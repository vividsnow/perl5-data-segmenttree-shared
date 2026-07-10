/*
 * segtree.h -- Shared-memory segment tree for Linux
 *
 * Range queries with range updates over a fixed array of n signed-integer
 * positions: add a delta to every element of a range in O(log n), and query the
 * sum, minimum, or maximum of any range in O(log n).  A perfect binary tree with
 * lazy propagation carries each node's sum/min/max plus a pending range-add; the
 * tree lives in a shared mapping so several processes update and query one array.
 * A write-preferring futex rwlock with reader-slot dead-process recovery guards
 * mutation; queries never mutate, so they take only the read lock.
 *
 * Layout: Header -> reader_slots[1024] -> nodes[2*next_pow2(n)]
 */

#ifndef ST_H
#define ST_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>
#include <limits.h>
#include <signal.h>
#include <stdio.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <sys/syscall.h>
#include <linux/futex.h>
#include <pthread.h>

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#error "segtree.h: requires little-endian architecture"
#endif


/* ================================================================
 * Constants
 * ================================================================ */

#define ST_MAGIC        0x54474553  /* SegmentTree */
#define ST_VERSION      1
#define ST_ERR_BUFLEN   256
#ifndef ST_READER_SLOTS
#define ST_READER_SLOTS 1024         /* max concurrent reader processes for dead-process recovery */
#endif
#define ST_MIN_N        1
#define ST_MAX_N        0x1000000ULL     /* 2^24 positions cap */
#define ST_IDENTITY_MIN INT64_MAX        /* min identity (padding leaves / empty range) */
#define ST_IDENTITY_MAX INT64_MIN        /* max identity */

#define ST_ERR(fmt, ...) do { if (errbuf) snprintf(errbuf, ST_ERR_BUFLEN, fmt, ##__VA_ARGS__); } while (0)

/* ================================================================
 * Structs
 * ================================================================ */

/* Per-process slot for dead-process recovery.  Each shared rwlock counter
 * (the main rwlock-reader count, rwlock_waiters, rwlock_writers_waiting)
 * is mirrored here so a wrlock timeout can attribute and reverse a dead
 * process's contribution instead of waiting for the slow per-op timeout
 * drain. */
typedef struct {
    uint32_t pid;            /* 0 = unclaimed */
    uint32_t subcount;       /* in-flight rdlock acquisitions for this process */
    uint32_t waiters_parked; /* contribution to hdr->rwlock_waiters         */
    uint32_t writers_parked; /* contribution to hdr->rwlock_writers_waiting */
} StReaderSlot;

struct StHeader {
    uint32_t magic, version;          /* 0,4 */
    uint32_t _pad0;                   /* 8 */
    uint32_t _pad1;                   /* 12 */
    uint64_t n;                       /* 16  number of positions (leaves in use) */
    uint64_t size;                    /* 24  next_pow2(n): leaves in the padded tree */
    uint64_t nodes_off;               /* 32  offset of the node array (2*size StNodes) */
    uint64_t total_size;              /* 40 */
    uint64_t reader_slots_off;        /* 48 */
    uint32_t rwlock;                  /* 56 */
    uint32_t rwlock_waiters;          /* 60 */
    uint32_t rwlock_writers_waiting;  /* 64 */
    uint32_t slotless_readers;  /* live readers holding the lock with no reader-slot */
    uint64_t stat_ops;                /* 72 */
    uint8_t  _pad[176];               /* 80..255 */
};
typedef struct StHeader StHeader;

_Static_assert(sizeof(StHeader) == 256, "StHeader must be 256 bytes");

/* One segment-tree node: the sum/min/max aggregate of its covered range, plus a
 * pending "range add" (lazy) delta not yet pushed to its children.  A node's own
 * sum/min/max already include its own lazy; only its children are stale by it. */
typedef struct {
    int64_t sum;
    int64_t min;
    int64_t max;
    int64_t lazy;
} StNode;
_Static_assert(sizeof(StNode) == 32, "StNode must be 32 bytes");

/* ---- Process-local handle ---- */

typedef struct StHandle {
    StHeader     *hdr;
    StReaderSlot *reader_slots;  /* ST_READER_SLOTS entries */
    void         *base;          /* mmap base */
    uint64_t      nodes_off;     /* validated, cached: never re-read from the peer-writable header */
    uint64_t      n;             /* cached number of positions */
    uint64_t      size;          /* cached next_pow2(n) */
    size_t        mmap_size;
    char         *path;          /* backing file path (strdup'd) */
    int           backing_fd;    /* memfd or reopened-fd to close on destroy, -1 for file/anon */
    uint32_t      my_slot_idx;   /* UINT32_MAX if all slots taken (no recovery for this handle) */
    uint32_t      cached_pid;    /* getpid() cached at last slot claim */
    uint32_t      cached_fork_gen; /* st_fork_gen value at last slot claim */
    uint32_t slotless_held; /* rwlock read-locks held with no reader-slot */
} StHandle;

/* ================================================================
 * Futex-based write-preferring read-write lock
 * with reader-slot dead-process recovery
 * ================================================================ */

#define ST_RWLOCK_SPIN_LIMIT 32
#define ST_LOCK_TIMEOUT_SEC  2  /* FUTEX_WAIT timeout for stale lock detection */

static inline void st_rwlock_spin_pause(void) {
#if defined(__x86_64__) || defined(__i386__)
    __asm__ volatile("pause" ::: "memory");
#elif defined(__aarch64__)
    __asm__ volatile("yield" ::: "memory");
#else
    __asm__ volatile("" ::: "memory");
#endif
}

/* Extract writer PID from rwlock value (lower 31 bits when write-locked). */
#define ST_RWLOCK_WRITER_BIT 0x80000000U
#define ST_RWLOCK_PID_MASK   0x7FFFFFFFU
#define ST_RWLOCK_WR(pid)    (ST_RWLOCK_WRITER_BIT | ((uint32_t)(pid) & ST_RWLOCK_PID_MASK))

/* Check if a PID is alive. Returns 1 if alive or unknown, 0 if definitely dead. */
/* Liveness via kill(pid,0). NOTE: cannot detect PID reuse -- if a dead
 * lock-holder's PID is recycled to an unrelated live process before recovery
 * runs, this reports "alive" and that slot's orphaned contribution is not
 * reclaimed until the recycled process exits. Robust detection would require
 * a per-slot process-start-time epoch (a header-layout/version change).
 * Documented under "Crash Safety" in the POD. */
static inline int st_pid_alive(uint32_t pid) {
    if (pid == 0) return 1; /* no owner recorded, assume alive */
    return !(kill((pid_t)pid, 0) == -1 && errno == ESRCH);
}

/* Force-recover a stale write lock left by a dead process.
 * CAS to OUR pid to hold the lock while fixing shared state, then release.
 * Using our pid (not a bare WRITER_BIT sentinel) means a subsequent
 * recovering process can detect and re-recover if we crash mid-recovery. */
static inline void st_recover_stale_lock(StHandle *h, uint32_t observed_rwlock) {
    StHeader *hdr = h->hdr;
    uint32_t mypid = ST_RWLOCK_WR((uint32_t)getpid());
    if (!__atomic_compare_exchange_n(&hdr->rwlock, &observed_rwlock,
            mypid, 0, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED))
        return;
    /* We now hold the write lock as mypid.  No additional shared state needs
     * repair here (this module has no seqlock); just release the lock. */
    __atomic_store_n(&hdr->rwlock, 0, __ATOMIC_RELEASE);
    if (__atomic_load_n(&hdr->rwlock_waiters, __ATOMIC_RELAXED) > 0)
        syscall(SYS_futex, &hdr->rwlock, FUTEX_WAKE, INT_MAX, NULL, NULL, 0);
}

static const struct timespec st_lock_timeout = { ST_LOCK_TIMEOUT_SEC, 0 };

/* Process-global fork-generation counter.  Incremented in the pthread_atfork
 * child callback so every open handle detects a fork transition on the next
 * lock call without paying a getpid() syscall on the hot path. */
static uint32_t st_fork_gen = 1;
static pthread_once_t st_atfork_once = PTHREAD_ONCE_INIT;
static void st_on_fork_child(void) {
    __atomic_add_fetch(&st_fork_gen, 1, __ATOMIC_RELAXED);
}
static void st_atfork_init(void) {
    pthread_atfork(NULL, NULL, st_on_fork_child);
}

/* Ensure this process owns a reader slot.  Called from the lock helpers so
 * that fork()'d children pick up their own slot lazily instead of sharing
 * the parent's.  Hot-path is a single relaxed load + compare; only on a
 * fork-generation mismatch do we touch getpid() and scan slots. */
static inline void st_claim_reader_slot(StHandle *h) {
    uint32_t cur_gen = __atomic_load_n(&st_fork_gen, __ATOMIC_RELAXED);
    if (__builtin_expect(cur_gen == h->cached_fork_gen && h->my_slot_idx != UINT32_MAX, 1))
        return;
    /* Cold path -- register the atfork hook once per process, then claim. */
    pthread_once(&st_atfork_once, st_atfork_init);
    /* Re-read after pthread_once: st_on_fork_child may have bumped it. */
    cur_gen = __atomic_load_n(&st_fork_gen, __ATOMIC_RELAXED);
    uint32_t now_pid = (uint32_t)getpid();
    h->cached_pid = now_pid;
    if (cur_gen != h->cached_fork_gen) h->slotless_held = 0;  /* fork: child holds none of the parent's slotless read locks */
    h->cached_fork_gen = cur_gen;
    h->my_slot_idx = UINT32_MAX;
    uint32_t start = now_pid % ST_READER_SLOTS;
    for (uint32_t i = 0; i < ST_READER_SLOTS; i++) {
        uint32_t s = (start + i) % ST_READER_SLOTS;
        uint32_t expected = 0;
        if (__atomic_compare_exchange_n(&h->reader_slots[s].pid,
                &expected, now_pid, 0,
                __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) {
            /* Zero all mirror fields, not just subcount: a SIGKILL'd
             * predecessor may have left waiters_parked/writers_parked
             * non-zero, and st_recover_dead_readers won't drain them
             * once we own the slot (the CAS expects the dead PID). */
            __atomic_store_n(&h->reader_slots[s].subcount, 0, __ATOMIC_RELAXED);
            __atomic_store_n(&h->reader_slots[s].waiters_parked, 0, __ATOMIC_RELAXED);
            __atomic_store_n(&h->reader_slots[s].writers_parked, 0, __ATOMIC_RELAXED);
            h->my_slot_idx = s;
            return;
        }
    }
    /* Table full -- leave my_slot_idx = UINT32_MAX so we silently skip
     * tracking for this handle (lock still works; just no recovery). */
}

/* Atomically subtract `sub` from a counter, capped at 0 (never underflows). */
static inline void st_atomic_sub_cap(uint32_t *p, uint32_t sub) {
    if (!sub) return;
    uint32_t cur = __atomic_load_n(p, __ATOMIC_RELAXED);
    for (;;) {
        uint32_t want = (cur > sub) ? cur - sub : 0;
        if (__atomic_compare_exchange_n(p, &cur, want,
                1, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
            return;
    }
}

/* Try to claim a dead slot (CAS pid -> 0) and drain its parked-waiter
 * contributions back to the global counters.  A no-op if the slot was stolen
 * by another recoverer or had no waiter contribution to drain.
 *
 * Note: subcount/waiters_parked/writers_parked are NOT zeroed here.
 * Between our CAS and a follow-up store, a new process could claim the
 * slot and start populating these fields -- our stores would clobber its
 * state.  st_claim_reader_slot zeros all three on every claim, so
 * leaving stale values is harmless. */
static inline void st_drain_dead_slot(StHandle *h, uint32_t i, uint32_t pid) {
    StHeader *hdr = h->hdr;
    uint32_t expected = pid;
    /* ACQ_REL on success: RELEASE publishes pid=0 to other observers;
     * ACQUIRE syncs us with prior writes from the dead process to
     * waiters_parked/writers_parked.  On weakly-ordered archs (aarch64)
     * a plain RELAXED load before the CAS could miss those writes;
     * loading them after the CAS keeps them inside the acquire window. */
    if (!__atomic_compare_exchange_n(&h->reader_slots[i].pid, &expected, 0,
            0, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
        return;
    uint32_t wp    = __atomic_load_n(&h->reader_slots[i].waiters_parked, __ATOMIC_RELAXED);
    uint32_t writp = __atomic_load_n(&h->reader_slots[i].writers_parked, __ATOMIC_RELAXED);
    if (wp)    st_atomic_sub_cap(&hdr->rwlock_waiters, wp);
    if (writp) st_atomic_sub_cap(&hdr->rwlock_writers_waiting, writp);
}

/* Scan reader slots for dead-process recovery.
 *
 * For each dead PID with non-zero contributions to the shared rwlock,
 * rwlock_waiters, or rwlock_writers_waiting counters, drain its share back
 * out so live processes don't have to wait for the slow per-op timeout
 * decrement to drain it for them.
 *
 * For the main rwlock counter we use the "no live reader holds -> force-
 * reset to 0" trick (precise) because per-process attribution of the
 * subcount is racy across the inc-counter-then-inc-subcount window. */
static inline void st_recover_dead_readers(StHandle *h) {
    if (!h->reader_slots) return;
    StHeader *hdr = h->hdr;
    int any_live_reader = 0;
    int found_dead_reader = 0;

    /* Pass 1: classify slots.  Slots with dead pid and sc == 0 (no rwlock
     * contribution to lose) are wiped immediately to free the slot for
     * future claimants and drain any orphan parked-waiter counters.  Slots
     * with dead pid and sc > 0 are left intact in this pass: if force-
     * reset cannot fire (because a live reader is concurrently present),
     * wiping the dead slot would lose the only record of its orphan
     * rwlock contribution and strand writers permanently once the live
     * reader releases. */
    for (uint32_t i = 0; i < ST_READER_SLOTS; i++) {
        uint32_t pid = __atomic_load_n(&h->reader_slots[i].pid, __ATOMIC_ACQUIRE);
        if (pid == 0) continue;
        uint32_t sc = __atomic_load_n(&h->reader_slots[i].subcount, __ATOMIC_RELAXED);
        if (st_pid_alive(pid)) {
            if (sc > 0) any_live_reader = 1;
            continue;
        }
        if (sc > 0) { found_dead_reader = 1; continue; }
        st_drain_dead_slot(h, i, pid);
    }

    /* Pass 2: only if force-reset will fire.  Issue the rwlock force-
     * reset CAS FIRST, while the window since pass 1's last scan is
     * still narrow (a handful of instructions, as in the original
     * single-pass code).  A new reader that started rdlock between
     * pass 1's scan and the CAS will either:
     *   (a) have already CAS'd rwlock from cur to cur+1 -- our CAS then
     *       fails (cur mismatched), recovery yields and a future
     *       cycle retries; or
     *   (b) be still in the subcount-bump phase -- our CAS sees the
     *       stale cur and resets to 0; the new reader's subsequent CAS
     *       rwlock(0 -> 1) succeeds cleanly.
     * Only after the CAS resolves do we wipe the deferred dead slots,
     * keeping that work outside the race-sensitive window. */
    /* A live reader with no slot (table was full) is invisible to the scan
     * above but still holds a +1 in the lock word; never force-reset under it. */
    if (__atomic_load_n(&hdr->slotless_readers, __ATOMIC_RELAXED) > 0)
        any_live_reader = 1;
    if (found_dead_reader && !any_live_reader) {
        /* ACQUIRE: a late reader's subcount++ (before its rwlock CAS) is then visible below. */
        uint32_t cur = __atomic_load_n(&hdr->rwlock, __ATOMIC_ACQUIRE);
        int drain_ok = 1;   /* keep dead slots if the reset doesn't fire */
        if (cur > 0 && cur < ST_RWLOCK_WRITER_BIT) {
            /* Re-scan for a live reader (fail-safe: only suppresses a reset). */
            int live_now = __atomic_load_n(&hdr->slotless_readers, __ATOMIC_RELAXED) > 0;
            for (uint32_t i = 0; !live_now && i < ST_READER_SLOTS; i++) {
                uint32_t p = __atomic_load_n(&h->reader_slots[i].pid, __ATOMIC_ACQUIRE);
                if (p && st_pid_alive(p) &&
                    __atomic_load_n(&h->reader_slots[i].subcount, __ATOMIC_RELAXED) > 0)
                    live_now = 1;
            }
            if (live_now) {
                drain_ok = 0;
            } else if (__atomic_compare_exchange_n(&hdr->rwlock, &cur, 0,
                    0, __ATOMIC_RELEASE, __ATOMIC_RELAXED)) {
                if (__atomic_load_n(&hdr->rwlock_waiters, __ATOMIC_RELAXED) > 0)
                    syscall(SYS_futex, &hdr->rwlock, FUTEX_WAKE, INT_MAX, NULL, NULL, 0);
            } else {
                drain_ok = 0;   /* rwlock changed under us -- shares may still be live */
            }
        }
        if (drain_ok) {
            for (uint32_t i = 0; i < ST_READER_SLOTS; i++) {
                uint32_t p = __atomic_load_n(&h->reader_slots[i].pid, __ATOMIC_ACQUIRE);
                if (p == 0 || st_pid_alive(p)) continue;
                st_drain_dead_slot(h, i, p);
            }
        }
    }
}

/* Inspect the lock word after a futex-wait timeout.  If a dead writer
 * holds it, force-recover the lock.  Otherwise drain dead readers' shares
 * of the rwlock/waiter counters.  Called from rdlock and wrlock ETIMEDOUT
 * branches -- identical recovery logic in both. */
static inline void st_recover_after_timeout(StHandle *h) {
    StHeader *hdr = h->hdr;
    uint32_t val = __atomic_load_n(&hdr->rwlock, __ATOMIC_RELAXED);
    if (val >= ST_RWLOCK_WRITER_BIT) {
        uint32_t pid = val & ST_RWLOCK_PID_MASK;
        if (!st_pid_alive(pid))
            st_recover_stale_lock(h, val);
    } else {
        st_recover_dead_readers(h);
    }
}

/* Park/unpark helpers: bump the global waiter counters together with this
 * process's mirrored slot counters so a wrlock-timeout recovery scan can
 * attribute and reverse a dead PID's contribution.  Kept paired to make
 * accidental drift between global and per-slot counts impossible. */
static inline void st_park_reader(StHandle *h) {
    if (h->my_slot_idx != UINT32_MAX)
        __atomic_add_fetch(&h->reader_slots[h->my_slot_idx].waiters_parked, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&h->hdr->rwlock_waiters, 1, __ATOMIC_RELAXED);
}
static inline void st_unpark_reader(StHandle *h) {
    __atomic_sub_fetch(&h->hdr->rwlock_waiters, 1, __ATOMIC_RELAXED);
    if (h->my_slot_idx != UINT32_MAX)
        __atomic_sub_fetch(&h->reader_slots[h->my_slot_idx].waiters_parked, 1, __ATOMIC_RELAXED);
}
static inline void st_park_writer(StHandle *h) {
    if (h->my_slot_idx != UINT32_MAX) {
        __atomic_add_fetch(&h->reader_slots[h->my_slot_idx].waiters_parked, 1, __ATOMIC_RELAXED);
        __atomic_add_fetch(&h->reader_slots[h->my_slot_idx].writers_parked, 1, __ATOMIC_RELAXED);
    }
    __atomic_add_fetch(&h->hdr->rwlock_waiters, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&h->hdr->rwlock_writers_waiting, 1, __ATOMIC_RELAXED);
}
static inline void st_unpark_writer(StHandle *h) {
    __atomic_sub_fetch(&h->hdr->rwlock_waiters, 1, __ATOMIC_RELAXED);
    __atomic_sub_fetch(&h->hdr->rwlock_writers_waiting, 1, __ATOMIC_RELAXED);
    if (h->my_slot_idx != UINT32_MAX) {
        __atomic_sub_fetch(&h->reader_slots[h->my_slot_idx].waiters_parked, 1, __ATOMIC_RELAXED);
        __atomic_sub_fetch(&h->reader_slots[h->my_slot_idx].writers_parked, 1, __ATOMIC_RELAXED);
    }
}

/* Reader accounting: a reader mirrors its +1 in the lock word so dead-reader
 * recovery can see it. A slotted reader uses its slot subcount; a reader that
 * could not claim a slot (table full) uses the global hdr->slotless_readers,
 * so recovery's force-reset never fires out from under it. leave() peels
 * slotless first so a later slot claim cannot misattribute the decrement. */
static inline void st_reader_enter(StHandle *h) {
    if (h->my_slot_idx != UINT32_MAX) {
        __atomic_add_fetch(&h->reader_slots[h->my_slot_idx].subcount, 1, __ATOMIC_RELAXED);
    } else {
        __atomic_add_fetch(&h->hdr->slotless_readers, 1, __ATOMIC_RELAXED);
        h->slotless_held++;
    }
}
static inline void st_reader_leave(StHandle *h) {
    if (h->slotless_held > 0) {
        h->slotless_held--;
        __atomic_sub_fetch(&h->hdr->slotless_readers, 1, __ATOMIC_RELAXED);
    } else if (h->my_slot_idx != UINT32_MAX) {
        __atomic_sub_fetch(&h->reader_slots[h->my_slot_idx].subcount, 1, __ATOMIC_RELAXED);
    }
}

static inline void st_rwlock_rdlock(StHandle *h) {
    st_claim_reader_slot(h);
    StHeader *hdr = h->hdr;
    uint32_t *lock = &hdr->rwlock;
    uint32_t *writers_waiting = &hdr->rwlock_writers_waiting;
    /* Claim subcount BEFORE bumping the shared rwlock counter.  This way
     * a concurrent writer-side recovery scan that sees our PID alive with
     * subcount > 0 will (correctly) defer force-reset, even while we are
     * still spinning trying to win the rwlock CAS.  Without this, a reader
     * killed between rwlock CAS-success and subcount++ would let recovery
     * force-reset rwlock to 0 underneath us, causing a UINT32_MAX wrap on
     * our eventual rdunlock dec. */
    st_reader_enter(h);
    for (int spin = 0; ; spin++) {
        uint32_t cur = __atomic_load_n(lock, __ATOMIC_RELAXED);
        /* Write-preferring: when lock is free (cur==0) and writers are
         * waiting, yield to let the writer acquire. When readers are
         * already active (cur>=1), new readers may join freely. */
        if (cur > 0 && cur < ST_RWLOCK_WRITER_BIT) {
            if (__atomic_compare_exchange_n(lock, &cur, cur + 1,
                    1, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
                return;
        } else if (cur == 0 && !__atomic_load_n(writers_waiting, __ATOMIC_RELAXED)) {
            if (__atomic_compare_exchange_n(lock, &cur, 1,
                    1, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
                return;
        }
        if (__builtin_expect(spin < ST_RWLOCK_SPIN_LIMIT, 1)) {
            st_rwlock_spin_pause();
            continue;
        }
        st_park_reader(h);
        cur = __atomic_load_n(lock, __ATOMIC_RELAXED);
        /* Sleep when write-locked OR when yielding to waiting writers */
        if (cur >= ST_RWLOCK_WRITER_BIT || cur == 0) {
            long rc = syscall(SYS_futex, lock, FUTEX_WAIT, cur,
                              &st_lock_timeout, NULL, 0);
            if (rc == -1 && errno == ETIMEDOUT) {
                st_unpark_reader(h);
                st_recover_after_timeout(h);
                spin = 0;
                continue;
            }
        }
        st_unpark_reader(h);
        spin = 0;
    }
}

static inline void st_rwlock_rdunlock(StHandle *h) {
    StHeader *hdr = h->hdr;
    /* Release the shared counter BEFORE dropping our subcount so that
     * "any live PID with subcount > 0" is a reliable in-flight indicator
     * for the writer-side recovery scan.  Inverting these would create a
     * window where we still own a unit of rwlock but our slot subcount is
     * 0, letting recovery force-reset rwlock underneath us. */
    uint32_t after = __atomic_sub_fetch(&hdr->rwlock, 1, __ATOMIC_RELEASE);
    st_reader_leave(h);
    if (after == 0 && __atomic_load_n(&hdr->rwlock_waiters, __ATOMIC_RELAXED) > 0)
        syscall(SYS_futex, &hdr->rwlock, FUTEX_WAKE, INT_MAX, NULL, NULL, 0);
}

static inline void st_rwlock_wrlock(StHandle *h) {
    st_claim_reader_slot(h);  /* refresh cached_pid across fork */
    StHeader *hdr = h->hdr;
    uint32_t *lock = &hdr->rwlock;
    /* Encode PID in the rwlock word itself (0x80000000 | pid) to eliminate
     * any crash window between acquiring the lock and storing the owner. */
    uint32_t mypid = ST_RWLOCK_WR(h->cached_pid);
    for (int spin = 0; ; spin++) {
        uint32_t expected = 0;
        if (__atomic_compare_exchange_n(lock, &expected, mypid,
                1, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED))
            return;
        if (__builtin_expect(spin < ST_RWLOCK_SPIN_LIMIT, 1)) {
            st_rwlock_spin_pause();
            continue;
        }
        st_park_writer(h);
        uint32_t cur = __atomic_load_n(lock, __ATOMIC_RELAXED);
        if (cur != 0) {
            long rc = syscall(SYS_futex, lock, FUTEX_WAIT, cur,
                              &st_lock_timeout, NULL, 0);
            if (rc == -1 && errno == ETIMEDOUT) {
                st_unpark_writer(h);
                st_recover_after_timeout(h);
                spin = 0;
                continue;
            }
        }
        st_unpark_writer(h);
        spin = 0;
    }
}

static inline void st_rwlock_wrunlock(StHandle *h) {
    StHeader *hdr = h->hdr;
    __atomic_store_n(&hdr->rwlock, 0, __ATOMIC_RELEASE);
    if (__atomic_load_n(&hdr->rwlock_waiters, __ATOMIC_RELAXED) > 0)
        syscall(SYS_futex, &hdr->rwlock, FUTEX_WAKE, INT_MAX, NULL, NULL, 0);
}

/* ================================================================
 * Layout math + create / open / destroy
 *
 * Layout: Header -> reader_slots[1024] -> nodes[2*next_pow2(n)]
 * ================================================================ */

/* Single source of truth for the mmap region layout offsets.
 * Layout: Header -> reader_slots[1024] -> nodes[2*size] (1-indexed, node 0 unused) */
typedef struct { uint64_t reader_slots, nodes, total; } StLayout;

/* round v up to the next power of two (64-bit), with a floor of 1 */
static inline uint64_t st_next_pow2_u64(uint64_t v) {
    if (v <= 1) return 1;
    return 1ULL << (64 - __builtin_clzll(v - 1));
}

static inline StLayout st_layout_for(uint64_t size) {
    StLayout L;
    L.reader_slots = sizeof(StHeader);
    L.nodes        = L.reader_slots + (uint64_t)ST_READER_SLOTS * sizeof(StReaderSlot);
    L.nodes        = (L.nodes + 7) & ~(uint64_t)7;
    L.total        = L.nodes + 2ULL * size * sizeof(StNode);   /* nodes 0 .. 2*size-1 */
    return L;
}

static inline uint64_t st_total_size(uint64_t size) {
    return st_layout_for(size).total;
}

static inline void st_init_header(void *base, uint64_t n, uint64_t size, uint64_t total) {
    StLayout L = st_layout_for(size);
    StHeader *hdr = (StHeader *)base;
    /* Zero the whole region: every node's sum/min/max/lazy = 0, so all n
       positions read as 0.  Padding leaves (n..size-1) stay 0 too -- no query
       ever covers a padding-containing node (queries clamp to [0, n-1]), so
       their aggregates never surface. */
    memset(base, 0, (size_t)L.total);
    hdr->magic            = ST_MAGIC;
    hdr->version          = ST_VERSION;
    hdr->n                = n;
    hdr->size             = size;
    hdr->nodes_off        = L.nodes;
    hdr->total_size       = total;
    hdr->reader_slots_off = L.reader_slots;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
}

static inline StNode *st_nodes(StHandle *h) {
    return (StNode *)((char *)h->base + h->nodes_off);
}

/* Layer B trusted bound: the number of nodes guaranteed to lie within the real
 * mapping.  Derived from the process-local mmap_size (fixed at attach, not
 * peer-writable) and the SAME nodes_off the accessor uses, so a corrupt
 * hdr->size can never drive a node access outside the mapping.  Equals 2*size
 * for a valid tree. */
static inline uint64_t st_nodes_max(StHandle *h) {
    if (h->nodes_off >= h->mmap_size) return 0;
    return (h->mmap_size - h->nodes_off) / sizeof(StNode);
}

static inline StHandle *st_setup(void *base, size_t map_size,
                                 const char *path, int backing_fd) {
    StHeader *hdr = (StHeader *)base;
    StHandle *h = (StHandle *)calloc(1, sizeof(StHandle));
    if (!h) {
        munmap(base, map_size);
        if (backing_fd >= 0) close(backing_fd);
        return NULL;
    }
    h->hdr          = hdr;
    h->base         = base;
    h->reader_slots = (StReaderSlot *)((uint8_t *)base + hdr->reader_slots_off);
    /* single validated read of each geometry field, cached process-locally */
    h->nodes_off    = hdr->nodes_off;
    h->n            = hdr->n;
    h->size         = hdr->size;
    h->mmap_size    = map_size;
    /* Layer B: if the mapping cannot hold 2*size nodes the header lied about its
       size; clamp the cached tree size to what actually fits. */
    {
        uint64_t fit = st_nodes_max(h) / 2;
        if (h->size > fit) { h->size = fit; if (h->n > h->size) h->n = h->size; }
    }
    h->path         = path ? strdup(path) : NULL;
    h->backing_fd   = backing_fd;
    h->my_slot_idx  = UINT32_MAX;
    return h;
}

/* Validate a mapped header (shared by st_create reopen and st_open_fd). */
static inline int st_validate_header(const StHeader *hdr, uint64_t file_size) {
    if (hdr->magic != ST_MAGIC) return 0;
    if (hdr->version != ST_VERSION) return 0;
    if (hdr->n < ST_MIN_N || hdr->n > ST_MAX_N) return 0;
    if (hdr->size != st_next_pow2_u64(hdr->n)) return 0;
    if (hdr->size == 0 || (hdr->size & (hdr->size - 1)) != 0) return 0;   /* power of two */
    if (hdr->total_size != file_size) return 0;
    if (hdr->total_size != st_total_size(hdr->size)) return 0;
    StLayout L = st_layout_for(hdr->size);
    if (hdr->reader_slots_off != L.reader_slots) return 0;
    if (hdr->nodes_off != L.nodes) return 0;
    return 1;
}

/* validate the requested number of positions n */
static int st_validate_args(uint64_t n, char *errbuf) {
    if (errbuf) errbuf[0] = '\0';
    if (n < ST_MIN_N || n > ST_MAX_N) { ST_ERR("number of positions must be between 1 and 2^24"); return 0; }
    return 1;
}

/* Securely obtain a fd for a path-backed segment: create it exclusively
 * (O_CREAT|O_EXCL|O_NOFOLLOW at `mode`, default 0600 = owner-only), or, if it
 * already exists, attach to it (O_RDWR|O_NOFOLLOW, no O_CREAT). O_EXCL blocks a
 * pre-seeded or hard-linked file and O_NOFOLLOW a symlink swap, so a local
 * attacker can no longer redirect or poison the backing store through the path.
 * Cross-user sharing is opt-in via a wider `mode` (e.g. 0660); the caller still
 * validates the file's contents via st_validate_header. */
static int st_secure_open(const char *path, mode_t mode, char *errbuf) {
    for (int attempt = 0; attempt < 100; attempt++) {
        int fd = open(path, O_RDWR|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC, mode);
        if (fd >= 0) { (void)fchmod(fd, mode); return fd; }   /* exact mode: umask narrowed the O_EXCL create */
        if (errno != EEXIST) { ST_ERR("create %s: %s", path, strerror(errno)); return -1; }
        fd = open(path, O_RDWR|O_NOFOLLOW|O_CLOEXEC);
        if (fd >= 0) return fd;
        if (errno == ENOENT) continue;   /* creator unlinked between our two opens; retry */
        ST_ERR("open %s: %s", path, strerror(errno));  /* ELOOP => symlink rejected */
        return -1;
    }
    ST_ERR("open %s: create/attach kept racing", path);
    return -1;
}

static StHandle *st_create(const char *path, uint64_t n, mode_t mode, char *errbuf) {
    if (!st_validate_args(n, errbuf)) return NULL;

    uint64_t size = st_next_pow2_u64(n);
    uint64_t total = st_total_size(size);
    int anonymous = (path == NULL);
    int fd = -1;
    size_t map_size;
    void *base;

    if (anonymous) {
        map_size = (size_t)total;
        base = mmap(NULL, map_size, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0);
        if (base == MAP_FAILED) { ST_ERR("mmap: %s", strerror(errno)); return NULL; }
    } else {
        fd = st_secure_open(path, mode, errbuf);
        if (fd < 0) return NULL;
        if (flock(fd, LOCK_EX) < 0) { ST_ERR("flock: %s", strerror(errno)); close(fd); return NULL; }
        struct stat st;
        if (fstat(fd, &st) < 0) { ST_ERR("fstat: %s", strerror(errno)); flock(fd, LOCK_UN); close(fd); return NULL; }
        int is_new = (st.st_size == 0);
        if (!is_new && (uint64_t)st.st_size < sizeof(StHeader)) {
            ST_ERR("%s: file too small (%lld)", path, (long long)st.st_size);
            flock(fd, LOCK_UN); close(fd); return NULL;
        }
        if (is_new && ftruncate(fd, (off_t)total) < 0) {
            ST_ERR("ftruncate: %s", strerror(errno)); flock(fd, LOCK_UN); close(fd); return NULL;
        }
        map_size = is_new ? (size_t)total : (size_t)st.st_size;
        base = mmap(NULL, map_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
        if (base == MAP_FAILED) { ST_ERR("mmap: %s", strerror(errno)); flock(fd, LOCK_UN); close(fd); return NULL; }
        if (!is_new) {
            if (!st_validate_header((StHeader *)base, (uint64_t)st.st_size)) {
                ST_ERR("invalid segment-tree file"); munmap(base, map_size); flock(fd, LOCK_UN); close(fd); return NULL;
            }
            flock(fd, LOCK_UN); close(fd);
            return st_setup(base, map_size, path, -1);
        }
    }
    st_init_header(base, n, size, total);
    if (fd >= 0) { flock(fd, LOCK_UN); close(fd); }
    return st_setup(base, map_size, path, -1);
}

static StHandle *st_create_memfd(const char *name, uint64_t n, char *errbuf) {
    if (!st_validate_args(n, errbuf)) return NULL;

    uint64_t size = st_next_pow2_u64(n);
    uint64_t total = st_total_size(size);
    int fd = memfd_create(name ? name : "segtree", MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (fd < 0) { ST_ERR("memfd_create: %s", strerror(errno)); return NULL; }
    if (ftruncate(fd, (off_t)total) < 0) {
        ST_ERR("ftruncate: %s", strerror(errno)); close(fd); return NULL;
    }
    (void)fcntl(fd, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_GROW);
    void *base = mmap(NULL, (size_t)total, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { ST_ERR("mmap: %s", strerror(errno)); close(fd); return NULL; }
    st_init_header(base, n, size, total);
    return st_setup(base, (size_t)total, NULL, fd);
}

static StHandle *st_open_fd(int fd, char *errbuf) {
    if (errbuf) errbuf[0] = '\0';
    struct stat st;
    if (fstat(fd, &st) < 0) { ST_ERR("fstat: %s", strerror(errno)); return NULL; }
    if ((uint64_t)st.st_size < sizeof(StHeader)) { ST_ERR("too small"); return NULL; }
    size_t ms = (size_t)st.st_size;
    void *base = mmap(NULL, ms, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { ST_ERR("mmap: %s", strerror(errno)); return NULL; }
    if (!st_validate_header((StHeader *)base, (uint64_t)st.st_size)) {
        ST_ERR("invalid segment-tree table"); munmap(base, ms); return NULL;
    }
    int myfd = fcntl(fd, F_DUPFD_CLOEXEC, 0);
    if (myfd < 0) { ST_ERR("fcntl: %s", strerror(errno)); munmap(base, ms); return NULL; }
    return st_setup(base, ms, NULL, myfd);
}

static void st_destroy(StHandle *h) {
    if (!h) return;
    /* Release our reader slot on clean teardown (else short-lived-reader churn
     * exhausts the slot table); skip if a lock is still held (subcount>0). */
    if (h->reader_slots && h->my_slot_idx != UINT32_MAX && h->cached_pid &&
        h->cached_fork_gen == __atomic_load_n(&st_fork_gen, __ATOMIC_RELAXED) &&
        __atomic_load_n(&h->reader_slots[h->my_slot_idx].subcount, __ATOMIC_ACQUIRE) == 0) {
        uint32_t expected = h->cached_pid;
        __atomic_compare_exchange_n(&h->reader_slots[h->my_slot_idx].pid,
                &expected, 0, 0, __ATOMIC_RELEASE, __ATOMIC_RELAXED);
    }
    if (h->backing_fd >= 0) close(h->backing_fd);
    if (h->base) munmap(h->base, h->mmap_size);
    free(h->path);
    free(h);
}

static inline int st_msync(StHandle *h) {
    if (!h || !h->base) return 0;
    return msync(h->base, h->mmap_size, MS_SYNC);
}

/* ================================================================
 * Segment-tree operations (callers hold the lock)
 *
 * A perfect binary tree over `size` = next_pow2(n) leaves; node 1 is the root
 * covering [0, size-1], node v has children 2v and 2v+1.  Positions n..size-1
 * are padding leaves that no query ever covers (queries are clamped to
 * [0, n-1]).  Each node keeps its subtree's sum/min/max plus a pending "range
 * add" delta (lazy) already reflected in its own aggregate but not yet in its
 * children's.  Range-add pushes lazy down on the way to recompute; queries never
 * mutate -- they carry the ancestors' un-pushed lazy as `pending` and shift each
 * covered node's aggregate by it.  All node indices come from the recursion
 * (never from shared memory), so the st_setup size-clamp alone keeps every access
 * inside the mapping.
 * ================================================================ */

/* apply a pending +delta to a node covering `cnt` leaves */
static inline void st_apply(StNode *nd, int64_t delta, uint64_t cnt) {
    nd->sum  += delta * (int64_t)cnt;
    nd->min  += delta;
    nd->max  += delta;
    nd->lazy += delta;
}

/* range-add delta to [l,r] within node v covering [lo,hi] (caller holds wrlock) */
static void st_range_add_rec(StNode *nodes, uint64_t v, uint64_t lo, uint64_t hi,
                             uint64_t l, uint64_t r, int64_t delta) {
    if (r < lo || hi < l) return;                              /* disjoint */
    if (l <= lo && hi <= r) { st_apply(&nodes[v], delta, hi - lo + 1); return; }  /* covered */
    uint64_t mid = lo + (hi - lo) / 2;
    int64_t lz = nodes[v].lazy;                               /* pushdown before recomputing */
    if (lz) {
        st_apply(&nodes[2*v],     lz, mid - lo + 1);
        st_apply(&nodes[2*v + 1], lz, hi - mid);
        nodes[v].lazy = 0;
    }
    st_range_add_rec(nodes, 2*v,     lo,      mid, l, r, delta);
    st_range_add_rec(nodes, 2*v + 1, mid + 1, hi,  l, r, delta);
    StNode *a = &nodes[2*v], *b = &nodes[2*v + 1];
    nodes[v].sum = a->sum + b->sum;
    nodes[v].min = a->min < b->min ? a->min : b->min;
    nodes[v].max = a->max > b->max ? a->max : b->max;
}

/* accumulate sum/min/max over [l,r] within node v covering [lo,hi], carrying the
 * ancestors' un-pushed lazy in `pending` (read-only; caller holds a lock) */
static void st_query_rec(StNode *nodes, uint64_t v, uint64_t lo, uint64_t hi,
                         uint64_t l, uint64_t r, int64_t pending,
                         int64_t *sum, int64_t *mn, int64_t *mx) {
    if (r < lo || hi < l) return;                             /* disjoint -> identity */
    if (l <= lo && hi <= r) {                                 /* covered */
        *sum += nodes[v].sum + pending * (int64_t)(hi - lo + 1);
        int64_t nmin = nodes[v].min + pending;
        int64_t nmax = nodes[v].max + pending;
        if (nmin < *mn) *mn = nmin;
        if (nmax > *mx) *mx = nmax;
        return;
    }
    uint64_t mid = lo + (hi - lo) / 2;
    int64_t newp = pending + nodes[v].lazy;                   /* carry v's un-pushed lazy down */
    st_query_rec(nodes, 2*v,     lo,      mid, l, r, newp, sum, mn, mx);
    st_query_rec(nodes, 2*v + 1, mid + 1, hi,  l, r, newp, sum, mn, mx);
}

/* clamp a caller range to [0, n-1]; returns 0 if it is empty/out of range */
static inline int st_clamp_range(StHandle *h, uint64_t *l, uint64_t *r) {
    if (h->n == 0 || h->size == 0) return 0;
    if (*l >= h->n) return 0;
    if (*r >= h->n) *r = h->n - 1;
    return *l <= *r;
}

/* add delta to every position in [l,r] (caller holds the write lock) */
static void st_range_add_locked(StHandle *h, uint64_t l, uint64_t r, int64_t delta) {
    if (!st_clamp_range(h, &l, &r)) return;
    st_range_add_rec(st_nodes(h), 1, 0, h->size - 1, l, r, delta);
}

/* sum/min/max over [l,r] into *sum/*mn/*mx (caller holds a lock) */
static void st_query_locked(StHandle *h, uint64_t l, uint64_t r,
                            int64_t *sum, int64_t *mn, int64_t *mx) {
    *sum = 0; *mn = ST_IDENTITY_MIN; *mx = ST_IDENTITY_MAX;
    if (!st_clamp_range(h, &l, &r)) return;
    st_query_rec(st_nodes(h), 1, 0, h->size - 1, l, r, 0, sum, mn, mx);
}

/* value at position i (caller holds a lock); 0 if out of range */
static int64_t st_get_locked(StHandle *h, uint64_t i) {
    int64_t s, mn, mx;
    st_query_locked(h, i, i, &s, &mn, &mx);
    return s;
}

/* set position i to val (caller holds the write lock) */
static void st_set_locked(StHandle *h, uint64_t i, int64_t val) {
    if (h->n == 0 || i >= h->n) return;
    int64_t cur = st_get_locked(h, i);
    int64_t delta = val - cur;
    if (delta) st_range_add_rec(st_nodes(h), 1, 0, h->size - 1, i, i, delta);
}

/* reset every position to 0 (caller holds the write lock) */
static inline void st_clear_locked(StHandle *h) {
    StNode *nodes = st_nodes(h);
    uint64_t node_count = 2 * h->size;
    uint64_t nmax = st_nodes_max(h);    /* Layer B: clamp to the mapping */
    if (node_count > nmax) node_count = nmax;
    memset(nodes, 0, (size_t)(node_count * sizeof(StNode)));   /* sum/min/max/lazy all 0 */
}

#endif /* ST_H */
