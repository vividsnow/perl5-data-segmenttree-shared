#define PERL_NO_GET_CONTEXT
#include "EXTERN.h"
#include "perl.h"
#include "XSUB.h"
#include "ppport.h"
#include "segtree.h"

#define EXTRACT(sv) \
    if (!sv_isobject(sv) || !sv_derived_from(sv, "Data::SegmentTree::Shared")) \
        croak("Expected a Data::SegmentTree::Shared object"); \
    StHandle *h = INT2PTR(StHandle*, SvIV(SvRV(sv))); \
    if (!h) croak("Attempted to use a destroyed Data::SegmentTree::Shared object")

#define MAKE_OBJ(class, handle) \
    SV *obj = newSViv(PTR2IV(handle)); \
    SV *ref = newRV_noinc(obj); \
    sv_bless(ref, gv_stashpv(class, GV_ADD)); \
    RETVAL = ref

/* validate + fetch a 0-based position, croaking on out-of-range (h->n is the
 * process-local, attach-validated position count -- not the peer-writable header) */
#define POS(nm, sv) \
    UV nm = SvUV(sv); \
    if (nm >= h->n) croak("Data::SegmentTree::Shared: position %" UVuf " out of range (n=%" UVuf ")", nm, (UV)h->n)

MODULE = Data::SegmentTree::Shared  PACKAGE = Data::SegmentTree::Shared

PROTOTYPES: DISABLE

SV *
new(class, path = &PL_sv_undef, n = 0, ...)
    const char *class
    SV *path
    UV n
  PREINIT:
    char errbuf[ST_ERR_BUFLEN];
  CODE:
    const char *p = (SvGETMAGIC(path), SvOK(path)) ? SvPV_nolen(path) : NULL;
    if (n < 1)
        croak("Data::SegmentTree::Shared->new: number of positions must be >= 1");
    /* Optional 4th arg: file mode for a newly-created file-backed segment
     * (default 0600, owner-only). Pass e.g. 0660 for cross-user sharing. */
    mode_t mode = (items > 3 && (SvGETMAGIC(ST(3)), SvOK(ST(3)))) ? (mode_t)SvUV(ST(3)) : 0600;
    StHandle *h = st_create(p, (uint64_t)n, mode, errbuf);
    if (!h) croak("Data::SegmentTree::Shared->new: %s", errbuf);
    MAKE_OBJ(class, h);
  OUTPUT:
    RETVAL

SV *
new_memfd(class, name = &PL_sv_undef, n = 0)
    const char *class
    SV *name
    UV n
  PREINIT:
    char errbuf[ST_ERR_BUFLEN];
  CODE:
    const char *nm = (SvGETMAGIC(name), SvOK(name)) ? SvPV_nolen(name) : NULL;   /* undef -> default label */
    if (n < 1)
        croak("Data::SegmentTree::Shared->new_memfd: number of positions must be >= 1");
    StHandle *h = st_create_memfd(nm, (uint64_t)n, errbuf);
    if (!h) croak("Data::SegmentTree::Shared->new_memfd: %s", errbuf);
    MAKE_OBJ(class, h);
  OUTPUT:
    RETVAL

SV *
new_from_fd(class, fd)
    const char *class
    int fd
  PREINIT:
    char errbuf[ST_ERR_BUFLEN];
  CODE:
    StHandle *h = st_open_fd(fd, errbuf);
    if (!h) croak("Data::SegmentTree::Shared->new_from_fd: %s", errbuf);
    MAKE_OBJ(class, h);
  OUTPUT:
    RETVAL

void
DESTROY(self)
    SV *self
  CODE:
    if (sv_isobject(self) && sv_derived_from(self, "Data::SegmentTree::Shared")) {
        StHandle *h = INT2PTR(StHandle*, SvIV(SvRV(self)));
        if (h) { sv_setiv(SvRV(self), 0); st_destroy(h); }   /* null first: activates EXTRACT's use-after-destroy croak + makes a double DESTROY a no-op */
    }

void
set(self, i, value)
    SV *self
    SV *i
    IV value
  PREINIT:
    EXTRACT(self);
  CODE:
    POS(pos, i);
    st_rwlock_wrlock(h);
    st_set_locked(h, (uint64_t)pos, (int64_t)value);
    __atomic_fetch_add(&h->hdr->stat_ops, 1, __ATOMIC_RELAXED);
    st_rwlock_wrunlock(h);

IV
add(self, i, delta)
    SV *self
    SV *i
    IV delta
  PREINIT:
    EXTRACT(self);
    int64_t v;
  CODE:
    POS(pos, i);
    st_rwlock_wrlock(h);
    st_range_add_rec(st_nodes(h), 1, 0, h->size - 1, (uint64_t)pos, (uint64_t)pos, (int64_t)delta);
    v = st_get_locked(h, (uint64_t)pos);   /* new value, under the same lock */
    __atomic_fetch_add(&h->hdr->stat_ops, 1, __ATOMIC_RELAXED);
    st_rwlock_wrunlock(h);
    RETVAL = (IV)v;
  OUTPUT:
    RETVAL

void
range_add(self, l, r, delta)
    SV *self
    SV *l
    SV *r
    IV delta
  PREINIT:
    EXTRACT(self);
  CODE:
    POS(lo, l);
    POS(hi, r);
    if (lo > hi) croak("Data::SegmentTree::Shared->range_add: l (%" UVuf ") > r (%" UVuf ")", lo, hi);
    st_rwlock_wrlock(h);
    st_range_add_locked(h, (uint64_t)lo, (uint64_t)hi, (int64_t)delta);
    __atomic_fetch_add(&h->hdr->stat_ops, 1, __ATOMIC_RELAXED);
    st_rwlock_wrunlock(h);

IV
get(self, i)
    SV *self
    SV *i
  PREINIT:
    EXTRACT(self);
    int64_t v;
  CODE:
    POS(pos, i);
    st_rwlock_rdlock(h);
    v = st_get_locked(h, (uint64_t)pos);
    st_rwlock_rdunlock(h);
    RETVAL = (IV)v;
  OUTPUT:
    RETVAL

# range aggregate helpers: each returns one value over [l, r]
IV
sum(self, l, r)
    SV *self
    SV *l
    SV *r
  PREINIT:
    EXTRACT(self);
    int64_t s, mn, mx;
  CODE:
    POS(lo, l);
    POS(hi, r);
    if (lo > hi) croak("Data::SegmentTree::Shared->sum: l > r");
    st_rwlock_rdlock(h);
    st_query_locked(h, (uint64_t)lo, (uint64_t)hi, &s, &mn, &mx);
    st_rwlock_rdunlock(h);
    RETVAL = (IV)s;
  OUTPUT:
    RETVAL

IV
min(self, l, r)
    SV *self
    SV *l
    SV *r
  PREINIT:
    EXTRACT(self);
    int64_t s, mn, mx;
  CODE:
    POS(lo, l);
    POS(hi, r);
    if (lo > hi) croak("Data::SegmentTree::Shared->min: l > r");
    st_rwlock_rdlock(h);
    st_query_locked(h, (uint64_t)lo, (uint64_t)hi, &s, &mn, &mx);
    st_rwlock_rdunlock(h);
    RETVAL = (IV)mn;
  OUTPUT:
    RETVAL

IV
max(self, l, r)
    SV *self
    SV *l
    SV *r
  PREINIT:
    EXTRACT(self);
    int64_t s, mn, mx;
  CODE:
    POS(lo, l);
    POS(hi, r);
    if (lo > hi) croak("Data::SegmentTree::Shared->max: l > r");
    st_rwlock_rdlock(h);
    st_query_locked(h, (uint64_t)lo, (uint64_t)hi, &s, &mn, &mx);
    st_rwlock_rdunlock(h);
    RETVAL = (IV)mx;
  OUTPUT:
    RETVAL

# query(l, r) -> { sum, min, max, count } over [l, r] in one lock
SV *
query(self, l, r)
    SV *self
    SV *l
    SV *r
  PREINIT:
    EXTRACT(self);
    int64_t s, mn, mx;
  CODE:
    POS(lo, l);
    POS(hi, r);
    if (lo > hi) croak("Data::SegmentTree::Shared->query: l > r");
    st_rwlock_rdlock(h);
    st_query_locked(h, (uint64_t)lo, (uint64_t)hi, &s, &mn, &mx);
    st_rwlock_rdunlock(h);
    {
        HV *hv = newHV();
        hv_stores(hv, "sum",   newSViv((IV)s));
        hv_stores(hv, "min",   newSViv((IV)mn));
        hv_stores(hv, "max",   newSViv((IV)mx));
        hv_stores(hv, "count", newSVuv((UV)(hi - lo + 1)));
        RETVAL = newRV_noinc((SV *)hv);
    }
  OUTPUT:
    RETVAL

void
clear(self)
    SV *self
  PREINIT:
    EXTRACT(self);
  CODE:
    st_rwlock_wrlock(h);
    st_clear_locked(h);
    __atomic_fetch_add(&h->hdr->stat_ops, 1, __ATOMIC_RELAXED);
    st_rwlock_wrunlock(h);

UV
size(self)
    SV *self
  PREINIT:
    EXTRACT(self);
  CODE:
    RETVAL = (UV)h->hdr->n;
  OUTPUT:
    RETVAL

SV *
stats(self)
    SV *self
  PREINIT:
    EXTRACT(self);
  CODE:
    {
        uint64_t n, sz, ops;
        n   = h->hdr->n;
        sz  = h->hdr->size;
        ops = __atomic_load_n(&h->hdr->stat_ops, __ATOMIC_RELAXED);
        HV *hv = newHV();
        hv_stores(hv, "n",         newSVuv((UV)n));
        hv_stores(hv, "size",      newSVuv((UV)n));
        hv_stores(hv, "tree_size", newSVuv((UV)sz));
        hv_stores(hv, "ops",       newSVuv((UV)ops));
        hv_stores(hv, "mmap_size", newSVuv((UV)h->mmap_size));
        RETVAL = newRV_noinc((SV *)hv);
    }
  OUTPUT:
    RETVAL

SV *
path(self)
    SV *self
  PREINIT:
    EXTRACT(self);
  CODE:
    RETVAL = h->path ? newSVpv(h->path, 0) : &PL_sv_undef;
  OUTPUT:
    RETVAL

int
memfd(self)
    SV *self
  PREINIT:
    EXTRACT(self);
  CODE:
    RETVAL = h->backing_fd;
  OUTPUT:
    RETVAL

void
sync(self)
    SV *self
  PREINIT:
    EXTRACT(self);
  CODE:
    if (st_msync(h) != 0) croak("sync: %s", strerror(errno));

void
unlink(self, ...)
    SV *self
  CODE:
    if (sv_isobject(self) && sv_derived_from(self, "Data::SegmentTree::Shared")) {
        StHandle *h = INT2PTR(StHandle*, SvIV(SvRV(self)));
        if (h && h->path) unlink(h->path);
    } else if (items >= 2 && (SvGETMAGIC(ST(1)), SvOK(ST(1)))) {
        unlink(SvPV_nolen(ST(1)));
    }
