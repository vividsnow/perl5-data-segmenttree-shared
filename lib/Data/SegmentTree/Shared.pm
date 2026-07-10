package Data::SegmentTree::Shared;
use strict;
use warnings;
our $VERSION = '0.01';
require XSLoader;
XSLoader::load('Data::SegmentTree::Shared', $VERSION);

sub CLONE_SKIP { 1 }  # blessed C-pointer handle: never clone into ithreads (double-free)
1;
__END__

=encoding utf-8

=head1 NAME

Data::SegmentTree::Shared - shared-memory segment tree (range-add, range sum/min/max)

=head1 SYNOPSIS

    use Data::SegmentTree::Shared;

    # an array of 1000 signed-integer positions, all 0
    my $st = Data::SegmentTree::Shared->new(undef, 1000);

    $st->set(10, 42);              # position 10 := 42
    $st->add(10, 5);               # position 10 += 5  (now 47)
    $st->range_add(0, 99, 3);      # add 3 to every position in [0, 99]

    my $s = $st->sum(0, 99);       # sum over a range
    my $lo = $st->min(0, 99);      # minimum over a range
    my $hi = $st->max(0, 99);      # maximum over a range
    my $q = $st->query(0, 99);     # { sum, min, max, count } in one call

    # share the array across processes via a backing file
    my $shared = Data::SegmentTree::Shared->new("/tmp/array.st", 1000);

=head1 DESCRIPTION

A B<segment tree> in shared memory: a fixed array of C<n> signed 64-bit integer
positions that supports B<range updates and range queries> in O(log n) each --
add a delta to every element of a range, and ask for the sum, minimum, or maximum
of any range. It complements L<Data::Fenwick::Shared> (which does prefix sums and
point updates): a segment tree adds B<range minimum and maximum> queries and
B<range add> (via lazy propagation), neither of which a Fenwick tree can do.

The tree is a perfect binary tree over C<next_pow2(n)> leaves; each node caches
its subtree's sum, min, and max, plus a pending "range add" delta that is pushed
down lazily. Range updates and queries therefore touch only O(log n) nodes.
Positions start at 0 and are addressed by a 0-based index; out-of-range indices
croak.

Because the tree lives in a shared mapping, B<several processes update and query
one array>: any process that opens the same backing file, inherits the anonymous
mapping across C<fork>, or reopens a passed memfd sees the same array. A
write-preferring futex rwlock with dead-process recovery guards mutation; queries
never mutate the tree, so they take only the read lock and many can run at once.
B<Linux-only>. Requires 64-bit Perl.

Values and range sums are signed 64-bit integers; feeding values large enough
that a range sum exceeds the 64-bit range overflows (wraps), as with any native
integer accumulator.

=head1 METHODS

=head2 Constructors

    my $st = Data::SegmentTree::Shared->new($path, $n, $mode);
    my $st = Data::SegmentTree::Shared->new(undef, $n);            # anonymous
    my $st = Data::SegmentTree::Shared->new_memfd($name, $n);
    my $st = Data::SegmentTree::Shared->new_from_fd($fd);

C<$n> is the number of positions (at least 1, up to 2^24); every position starts
at 0. Memory is C<2 * next_pow2(n) * 32> bytes plus a fixed header. C<new> and
C<new_memfd> croak on a C<$n> below 1 or above 2^24. When reopening an existing
file or memfd the stored C<$n> wins and the caller's argument is ignored. An
optional file B<mode> may be passed as the last argument to C<new> (e.g. C<0660>)
for cross-user sharing; it defaults to C<0600> (owner-only).

=head2 Updates

    $st->set($i, $value);          # position $i := $value
    my $new = $st->add($i, $delta); # position $i += $delta; returns the new value
    $st->range_add($l, $r, $delta); # add $delta to every position in [$l, $r]

C<set> assigns a single position; C<add> adds a delta to a single position and
returns its new value; C<range_add> adds a delta to every position in the
inclusive range C<[$l, $r]> in O(log n) (its defining feature). All indices are
0-based and croak if out of range; C<range_add> croaks if C<$l > $r>.

=head2 Queries

    my $v = $st->get($i);          # value at position $i
    my $s = $st->sum($l, $r);      # sum over [$l, $r]
    my $lo = $st->min($l, $r);     # minimum over [$l, $r]
    my $hi = $st->max($l, $r);     # maximum over [$l, $r]
    my $q = $st->query($l, $r);    # { sum, min, max, count } in one locked call

C<get> returns a single position's value. C<sum>, C<min>, and C<max> return one
aggregate over the inclusive range C<[$l, $r]>. C<query> returns all of them at
once as a hash reference C<< { sum, min, max, count } >> (C<count> is
C<$r - $l + 1>), computed under a single read lock so the four values are
mutually consistent. Ranges croak if an index is out of range or C<$l > $r>.

=head2 Introspection and lifecycle

    $st->size;          # n, the number of positions
    $st->clear;         # reset every position to 0
    $st->stats;         # { n, size, tree_size, ops, mmap_size }
    $st->path; $st->memfd; $st->sync; $st->unlink;

C<clear> resets every position to 0. C<sync> flushes the mapping to its backing
store (a no-op for anonymous and memfd trees); C<unlink> removes the backing file
(also callable as C<< Class->unlink($path) >>); C<path> returns the backing path
(C<undef> for anonymous, memfd, or fd-reopened trees) and C<memfd> the backing
descriptor.

=head1 SHARING ACROSS PROCESSES

The tree lives in a shared mapping, shared the same three ways as the rest of the
family: a B<backing file>, an B<anonymous mapping inherited across C<fork>>, or a
B<memfd> passed to an unrelated process and reopened with C<< new_from_fd($fd) >>.
Every process's updates land in the one shared array, and queries take only the
read lock so many readers proceed concurrently.

=head1 SECURITY

Backing files are created with mode C<0600> (owner-only) by default; pass an
explicit octal mode (e.g. C<0660>) as the last argument to C<new> for cross-user
sharing. The file is opened with C<O_NOFOLLOW> and C<O_EXCL>, and the header is
validated on attach. Any process granted write access is trusted not to corrupt
the mapping.

=head1 CRASH SAFETY

Mutation is guarded by a futex-based write-preferring rwlock with PID-encoded
ownership and dead-owner recovery. Each update is a short bounded O(log n) tree
walk, so a crash leaves the array consistent up to the last completed operation.
B<Limitation>: PID reuse is not detected (very unlikely in practice).

=head1 SEE ALSO

L<Data::Fenwick::Shared> (prefix sums / point updates), and the rest of the
C<Data::*::Shared> family.

=head1 AUTHOR

vividsnow

=head1 LICENSE

This is free software; you can redistribute it and/or modify it under the same
terms as Perl itself.

=cut
