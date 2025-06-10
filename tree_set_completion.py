#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Aleksandr Koshkarov
# Version: May 2025

"""
Tree-set completion algorithm.

 – Reads each multiset  input_multisets/multiset_*.txt
 - Constructs consensus MCS
 – Inserts each consensus MCS to complete a tree
 – Writes completed sets to completed_multisets/completed_multiset_i.txt
"""

# ───────────────────────── import ──────────────────────────
import os, re, math, argparse, glob
from collections import defaultdict
import numpy  as np
from ete3 import Tree

# These are used by compute_weights_globally() and some helpers
global_T_set = []          # store the current list of host trees
global_T_i   = None        # store the current target tree

# ───────────────────────────── CLI ───────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=None,
                help="k for the number of common leaves (default = |common|)")
args = ap.parse_args()
K_USER = args.k

#
# Section 1. Utility functions
#

def preprocess_newick(s):            # spaces to underscores inside labels
    return re.sub(r'(?<=\w) (?=\w)', '_', s)

def get_leaf_set_ete(t):             # returns set of leaf names
    return {lf.name for lf in t.get_leaves() if lf.name}

def is_rooted_ete(t):
    root = t
    while root.up: root = root.up
    return len(root.children) <= 2

def midpoint_root(t):                # root a nearly unrooted tree
    try:
        og = t.get_midpoint_outgroup()
        if og != t: t.set_outgroup(og)
    except: pass

def node_to_ancestor_distance(node, anc):
    d = 0.0
    cur = node
    while cur and cur != anc:
        d += cur.dist if cur.dist else 0
        cur = cur.up
    return d

def assign_internal_node_names(t, prefix="INODE"):
    """Make internal node names unique for debugging / DistOracle."""
    idx = 1
    for n in t.traverse("preorder"):
        if not n.is_leaf():
            n.name = f"{prefix}_{idx}"; idx += 1

def clear_internal_node_names(t):
    for n in t.traverse():
        if not n.is_leaf(): n.name = ""

# 
# Section 2.Distance oracle & optimization helpers 
#

from functools import lru_cache
from math       import floor, log2

class DistOracle:
    """O(1) distance queries between any two nodes after O(n log n) build."""
    __slots__ = ("dist_to_root","_euler","_depth","_first","_st","_log2",
                 "_dist_cached")

    def __init__(self, tree: Tree, cache_size=20_000):
        self.dist_to_root = {}
        self._annotate_depth(tree)
        self._build_lca_struct(tree)
        @lru_cache(maxsize=cache_size)
        def _dc(a, b):
            lca = self._lca(a, b)
            return (self.dist_to_root[a] +
                    self.dist_to_root[b] -
                    2 * self.dist_to_root[lca])
        self._dist_cached = _dc

    def dist(self, a, b): return self._dist_cached(a, b)
    def dist_leaf_to_node(self, leaf, node): return self.dist(leaf, node)

    def _annotate_depth(self, tree):
        for n in tree.traverse("preorder"):
            self.dist_to_root[n] = 0.0 if n.up is None else \
                                   self.dist_to_root[n.up] + n.dist
    def _build_lca_struct(self, tree):
        eul, dep, first = [], [], {}
        def dfs(v,d):
            first.setdefault(v, len(eul))
            eul.append(v); dep.append(d)
            for ch in v.children:
                dfs(ch,d+1); eul.append(v); dep.append(d)
        dfs(tree,0)
        self._euler,self._depth,self._first=eul,dep,first
        m=len(eul); kmax=floor(log2(m))+1
        st=[[0]*m for _ in range(kmax)]
        st[0]=list(range(m))
        for k in range(1,kmax):
            half=1<<(k-1)
            for i in range(m-(1<<k)+1):
                a,b=st[k-1][i],st[k-1][i+half]
                st[k][i]= a if dep[a]<dep[b] else b
        self._st=st
        self._log2=[0]*(m+1)
        for i in range(2,m+1):
            self._log2[i]=self._log2[i>>1]+1
    def _lca(self,a,b):
        ia,ib=self._first[a],self._first[b]
        if ia>ib: ia,ib=ib,ia
        span=ib-ia+1
        k=self._log2[span]
        left=self._st[k][ia]
        right=self._st[k][ib-(1<<k)+1]
        return self._euler[left] if self._depth[left]<self._depth[right] \
               else self._euler[right]


def scale_subtree(root, factor):
    for n in root.traverse(): n.dist *= factor

def find_distinct_leaves(T, common):       # leaves unique to T
    return get_leaf_set_ete(T) - common


def find_optimal_insertion_point(target_tree, Dtgt,
                                 subtree_root,
                                 ncl_names, d_p,
                                 dl_names,             # distinct leaves (DLs)
                                 min_terminal=1e-3,
                                 eps=1e-6):
    """
    Objective function quadratic optimization.

    Parameters
    ----------
    target_tree : ete3.Tree      (tree being modified)
    Dtgt        : DistOracle     (distance oracle on current target_tree)
    subtree_root: ete3.TreeNode  (kept for backward compatibility)
    ncl_names   : list[str]      common leaves
    d_p         : dict           target distances after leaf–based scaling
    dl_names    : set[str]       "distinct leaves" (anchors must avoid edges already containing any of these)
    min_terminal: float          minimum length to leave on each side
    eps         : float          tolerance for "almost parent/child"
    """
    best_edge, best_x, best_val = None, 0.0, float("inf")
    dl_names   = set(dl_names)

    for ch in target_tree.traverse("postorder"):
        par = ch.up
        if par is None:
            continue                               # root has no parent edge
        

        # skip edges that are fully inside a previously inserted subtree
        if getattr(ch,  "inInserted", False):
            continue

        # avoid edges whose subtree already contains a DL
        if set(ch.get_leaf_names()) & dl_names:
            continue

        elen = ch.dist
        if elen < 1e-15:     # effectively zero length
            continue
            
        # proper-partition check  (must split the anchor set)
        ncl_set    = set(ncl_names)
        below = set(ch.get_leaf_names()) & ncl_set
        if len(below) == 0 or len(below) == len(ncl_names):
            continue                           # anchors all on one side - skip

        m             = len(ncl_names)
        ch_leaf_set   = set(ch.get_leaf_names())   # cache per edge
     
        # ----- closed-form optimum  ---------------------------
        num = 0.0
        for n in ncl_names:
            sgn  = -1 if n in ch_leaf_set else +1 
            num += sgn * (d_p[n] -
                          Dtgt.dist_leaf_to_node(target_tree & n, par))
        x_opt = num / (elen * m)

        # clamp x into the valid range [0,1)
        if ch.is_leaf():          # cannot split within a terminal branch
            x_opt = max(0.0, min(x_opt, 1 - min_terminal / elen))
        else:
            x_opt = max(0.0, min(x_opt, 1.0 - eps))

        # ----- evaluate OF for candidate x’s ------------------------------
        for x in (x_opt, 0.0, 1.0 - eps):
            of = 0.0
            for n in ncl_names:
                sgn = -1 if n in ch_leaf_set else +1
                obs = Dtgt.dist_leaf_to_node(target_tree & n, par) + sgn * x * elen
                diff = obs - d_p[n]
                of  += diff * diff
            if of < best_val - 1e-12:
                best_edge, best_x, best_val = (par, ch), x, of
          
    return best_edge, best_x, best_val


def insert_subtree_at_point(target_tree, edge, x_opt, subtree_copy,
                            eps=1e-6, min_terminal=1e-9):
    parent, child = edge
    orig_len = child.dist

    # 1. almost-parent
    if x_opt <= eps:
        parent.add_child(subtree_copy)
        return

    # 2. almost-child
    if x_opt >= 1 - eps:
        # if child is internal, we can safely insert the subtree under it
        if not child.is_leaf():
            child.add_child(subtree_copy)
            return
        # if child is a LEAF, we must split the edge to keep the leaf label
        d_up   = max(orig_len - min_terminal, min_terminal)
        d_down = min_terminal
        mid = Tree(); mid.dist = d_up
        mid.add_feature("inInserted", True)

        parent.remove_child(child)
        parent.add_child(mid)
        child.dist = d_down
        mid.add_child(child)          # keep the original leaf
        mid.add_child(subtree_copy)   # attach the new subtree
        assign_internal_node_names(target_tree)
        return

    # 3. genuine split
    d_up   = max(x_opt * orig_len,       min_terminal)
    d_down = max((1 - x_opt) * orig_len, min_terminal)

    mid = Tree(); mid.dist = d_up
    parent.remove_child(child)
    parent.add_child(mid)
    child.dist = d_down
    mid.add_child(child)
    mid.add_child(subtree_copy)
    mid.add_feature("inInserted", True)
    assign_internal_node_names(target_tree)

# 
# Section 3. Average-distance helpers across host trees
#

def compute_leaf_lca_distance_from_hosts(c, host_trees, subtree_leaves):
    """Average distance from leaf c to MRCA(subtree_leaves) over host_trees."""
    s=sum_d=0.0
    needed=list(subtree_leaves)
    for Th in host_trees:
        leaves=Th.get_leaf_names()
        if c not in leaves or not set(needed).issubset(leaves): continue
        try:
            mrca=Th.get_common_ancestor(needed)
            sum_d+=Th.get_distance(c,mrca); s+=1
        except: continue
    return 0.0 if s==0 else sum_d/s

def compute_global_adjustment_rate(T_tgt, host_trees, common_names):
    """averaged over all host trees."""
    if len(common_names)<2: return 1.0
    # distance in target
    pairs=[(a,b) for i,a in enumerate(common_names)
                  for b in list(common_names)[i+1:]]
    num=den=0.0
    for a,b in pairs:
        num+=T_tgt.get_distance(a,b)
        s=cnt=0.0
        for Th in host_trees:
            if a in Th.get_leaf_names() and b in Th.get_leaf_names():
                s+=Th.get_distance(a,b); cnt+=1
        if cnt: den+=s/cnt
    return 1.0 if abs(den)<1e-15 else num/den

def compute_local_leaf_based_rate(T_tgt, host_trees, c, common):
    """averaged across hosts."""
    others=list(common-{c})
    if not others: return 1.0
    #num=sum(T_tgt.get_distance(c,o) for o in others)
    num = sum(T_tgt.get_distance(c, o) for o in others) / len(others)
    den=cnt=0.0
    for Th in host_trees:
        if c not in Th.get_leaf_names(): continue
        s=l=0.0
        for o in others:
            if o in Th.get_leaf_names():
                s+=Th.get_distance(c,o); l+=1
        if l: den+=s/l; cnt+=1
    return 1.0 if cnt==0 or abs(den)<1e-15 else num/(den/cnt)

#
# Section 4.  Insertion of one consensus subtree
#

def insert_subtree_kncl(target_tree, original_target,
                        subtree_star, host_trees,
                        anchor_leaves_original,
                        k_user=None):
    """
    • Scales subtree_star by global τ
    • Finds common leaves (avg distances over host trees)
    • Solves quadratic OF to inserts subtree
    Returns True on success.
    """
    anchor_leaves_original=set(anchor_leaves_original)
    if len(anchor_leaves_original)<2: return False

    # 0. prepare names and distance oracle on current target tree
    assign_internal_node_names(target_tree,"TGT")
    Dtgt = DistOracle(target_tree)

    # 1. global-scale factor  τ
    τ = compute_global_adjustment_rate(original_target,
                                       host_trees,
                                       anchor_leaves_original)

    # 2. copy & scale subtree
    S = subtree_star.copy(method="deepcopy")
    scale_subtree(S, τ)
    for n in S.traverse(): n.add_feature("inInserted", True)
 
    subtree_leaves = get_leaf_set_ete(S)
    
    #  single-leaf vs. multi-leaf handling
    if len(subtree_leaves) == 1:
        # --- SINGLE-LEAF CASE ---
        # 1. grab the only child (= real leaf)
        leaf = next(iter(S.children))
     
        # 2.  separate it from the dummy root so it has no other parent
        leaf.detach()
        S = S_root = leaf
        attach_len = 0.0            # no connector edge for single leaves
        leaf.add_feature("inInserted", True)
    else:
        S_root = S               # multi-leaf: keep the small subtree
        raw = getattr(subtree_star, "connecting_length", 1.0)
        attach_len = max(raw * τ, 1e-9)
        S.dist = attach_len      # give the root its connector length
        
    # Mark every node inside the new subtree (root included)
    S_root.add_feature("inInserted", True)
    for n in S_root.traverse():
        n.add_feature("inInserted", True)
    
    # 3. average distances leaf and MRCA(subtree_leaves) across hosts        
    d_avg = {}
    for c in anchor_leaves_original:
        if c in subtree_leaves:          # anchor is inside S  - scale by τ
            try:
                d_avg[c] = τ * subtree_star.get_distance(
                               c, subtree_star.get_common_ancestor(list(subtree_leaves)))
            except:
                d_avg[c] = 0.0
        else:                            # anchor is outside S - no τ-scaling
            d_avg[c] = compute_leaf_lca_distance_from_hosts(
                           c, host_trees, subtree_leaves)
        
    # 4. choose k
    k_default = len(anchor_leaves_original)
    k = k_user if (k_user and k_user>=2) else k_default
    pairs = sorted(d_avg.items(), key=lambda t: t[1])
    kth = pairs[min(k, len(pairs))-1][1] if pairs else 0.0
    ncl = [name for name,d in pairs if d <= kth+1e-15]
    if len(ncl)<2: return False        # need >= 2 anchors for optimization

    # 5.  target distances d_p  (leaf-based scaling ρ_c)
    d_p = {}
    for c in ncl:
        ρ = compute_local_leaf_based_rate(original_target,
                                          host_trees, c,
                                          anchor_leaves_original)
        d_p[c] = d_avg[c] * ρ

    # 6.  quadratic optimum
    edge,x_opt,_ = find_optimal_insertion_point(
        target_tree, Dtgt, S_root, ncl, d_p,
        find_distinct_leaves(target_tree, anchor_leaves_original)
    )
    if not edge:                                 # Fallback
        target_tree.add_child(S, dist=attach_len)
        edge = ("root", "root")       # mark for logging
        x_opt = 0.0
    else:                                         # Normal insert
        insert_subtree_at_point(target_tree, edge, x_opt, S)
    
    return True

#
# Section 5. High-level tree-set completion part
#

def get_common_leaves(trees):
    if not trees: return set()
    common = get_leaf_set_ete(trees[0])
    for t in trees[1:]: common &= get_leaf_set_ete(t)
    return common

def extract_splits(tree):
    splits = []
    total_leaves = len(get_leaf_set_ete(tree))
    for nd in tree.traverse("postorder"):
        if nd.is_leaf() or nd.up is None:
            continue
        leaves_under = {x.name for x in nd.get_leaves() if x.name}
        if len(leaves_under) <= total_leaves / 2:
            splits.append(frozenset(leaves_under))
    return splits

def compute_weights_globally():
    if not global_T_set or global_T_i is None:
        return {}
    L_Ti = get_leaf_set_ete(global_T_i)
    denom = len(L_Ti)
    wt = {}
    for idx, T_j in enumerate(global_T_set):
        L_Tj = get_leaf_set_ete(T_j)
        ov = len(L_Tj & L_Ti)
        w_j = ov/denom if denom > 0 else 0
        wt[idx] = w_j
    return wt

def build_consensus_topology_from_splits_and_root(leaves, majority_splits, best_root_clade):
    t = Tree();  t.name = "WeightedConsensus";  t.dist = 0.0
    if not leaves:
        return t
    if len(leaves) == 1:
        t.add_child(name=list(leaves)[0], dist=1.0)
        return t

    for label in leaves:
        t.add_child(name=label, dist=1.0)

    majority_splits_sorted = sorted(majority_splits, key=lambda s: len(s), reverse=True)
    for sp in majority_splits_sorted:
        try:
            mrca = t.get_common_ancestor(list(sp))
            if mrca and mrca.up:
                mrca.dist = 1.0
        except:
            continue

    all_leaves = {lf.name for lf in t.get_leaves() if lf.name}
    if best_root_clade and 0 < len(best_root_clade) < len(all_leaves):
        outside = list(all_leaves - set(best_root_clade))
        if outside:
            out_label = outside[0]
            out_node  = t.search_nodes(name=out_label)
            if out_node:
                try:
                    t.set_outgroup(out_node[0])
                except:
                    pass

    for nd in t.traverse("postorder"):
        if nd.up and (nd.dist is None or nd.dist <= 0):
            nd.dist = 1e-9
    return t

def build_weighted_consensus_topology(S_list, p, T_i):
    all_leaves, root_clade_counts, split_counts = set(), defaultdict(float), defaultdict(float)
    w_j = compute_weights_globally();     sum_w = sum(w_j.values())

    for (S_j, _, host_idx, _) in S_list:
        leaves_j = get_leaf_set_ete(S_j)
        all_leaves.update(leaves_j)
        root_clade_counts[frozenset(leaves_j)] += w_j[host_idx]
        for sp in extract_splits(S_j):
            split_counts[sp] += w_j[host_idx]

    split_freq = {sp: (wsum/sum_w if sum_w else 0) for sp, wsum in split_counts.items()}
    majority_splits = [sp for sp, freq in split_freq.items() if freq > p]

    root_freq = {rc: (wsum/sum_w if sum_w else 0) for rc, wsum in root_clade_counts.items()}
    best_root = max(root_freq, key=root_freq.get, default=None)

    return build_consensus_topology_from_splits_and_root(all_leaves, majority_splits, best_root), \
           sorted({host_idx for (_, _, host_idx, _) in S_list})

# branch lengths

def least_squares_fit_branch_lengths(consensus_tree, avg_dist):
    if not is_rooted_ete(consensus_tree):
        midpoint_root(consensus_tree)

    for n in consensus_tree.traverse("postorder"):
        if n.dist is None or n.dist <= 0:
            n.dist = 1e-9

    edges, node2idx = [], {}
    for idx, nd in enumerate(consensus_tree.traverse("postorder")):
        if nd.up is not None:
            edges.append((nd, nd.up))
            node2idx[nd] = idx

    leaves = sorted(get_leaf_set_ete(consensus_tree))
    leaf_map = {lf.name: lf for lf in consensus_tree.get_leaves() if lf.name}
    M, d_vec = [], []
    for (la, lb), dval in avg_dist.items():
        if la not in leaf_map or lb not in leaf_map: continue
        a, b = leaf_map[la], leaf_map[lb]
        path_a, cur = [], a
        while cur and cur.up: path_a.append(cur); cur = cur.up
        path_b, cur = [], b
        while cur and cur.up: path_b.append(cur); cur = cur.up
        while path_a and path_b and path_a[-1] == path_b[-1]:
            path_a.pop(); path_b.pop()
        row = np.zeros(len(edges))
        for nd in path_a+path_b:
            if nd in node2idx: row[node2idx[nd]] += 1.0
        M.append(row);  d_vec.append(dval)

    M, d_vec = np.array(M), np.array(d_vec)
    if M.size and M.shape[1]:
        try:
            x, *_ = np.linalg.lstsq(M, d_vec, rcond=None)
            for nd, idx in node2idx.items():
                nd.dist = max(x[idx], 1e-9)
        except np.linalg.LinAlgError:
            pass
    return consensus_tree

def compute_average_distances(S_list, wt):
    alls=set().union(*(get_leaf_set_ete(S_j) for (S_j,_,_,_) in S_list))
    alls=sorted(alls)
    dist_sum={(alls[i], alls[j]): 0.0
              for i in range(len(alls)) for j in range(i+1,len(alls))}
    sum_w = sum(wt.values())
    for (S_j, _, host_idx, _ ) in S_list:
        w = wt.get(host_idx, 0.0)
        lset = get_leaf_set_ete(S_j)
        for i in range(len(alls)):
            for j in range(i+1,len(alls)):
                la, lb = alls[i], alls[j]
                if la in lset and lb in lset:
                    try:
                        mm = S_j.get_common_ancestor(la, lb)
                        da = node_to_ancestor_distance(S_j&la, mm)
                        db = node_to_ancestor_distance(S_j&lb, mm)
                        dist_sum[(la,lb)] += w*(da+db)
                    except: pass
    avg_dist={(la,lb):(v/sum_w if sum_w else 0.0)
              for (la,lb),v in dist_sum.items()}
    return avg_dist, alls


def extract_clusters(tree, U, multi_leaf=True):
    clusters=[]
    for nd in tree.traverse("postorder"):
        if nd.is_leaf():
            if not multi_leaf and nd.name in U:
                clusters.append(frozenset([nd.name]))
            continue
        leaves_under={x.name for x in nd.get_leaves() if x.name}
        if leaves_under and leaves_under.issubset(U):
            if multi_leaf and len(leaves_under)>1:
                clusters.append(frozenset(leaves_under))
            elif not multi_leaf and len(leaves_under)==1:
                clusters.append(frozenset(leaves_under))
    return clusters

def frequency_and_filter(C, T_set, p, wt):
    sum_w=sum(wt[idx] for idx in range(len(T_set)))
    C_freq={c: sum(wt[idx] for idx,Tj in enumerate(T_set)
                   if c.issubset(get_leaf_set_ete(Tj))) / sum_w
            for c in C} if sum_w else {c:0.0 for c in C}
    C_p=[c for c in C if C_freq[c]>p]
    return C_p, C_freq

def group_clusters_by_leafset(C_p):
    groups=defaultdict(list)
    for c in C_p: groups[frozenset(c)].append(c)
    return list(groups.values())

def max_coverage_group(groups, U):
    best = None         # group that gives max coverage
    best_cov = -1       # numeric coverage value
    for G in groups:
        leaf_set = set().union(*G)
        cv = len(leaf_set & U)
        if cv > best_cov:
            best_cov = cv
            best     = G
    return best

def extract_subtree(C, T):
    if not C: return None
    copyT = T.copy(method='deepcopy')
    leaves_to_prune = set(x.name for x in copyT.get_leaves() if x.name)-C
    for lf in leaves_to_prune:
        for nd in copyT.search_nodes(name=lf): nd.detach()
    for nd in copyT.traverse("postorder"):
        nd.dist = max(nd.dist or 0.0, 1e-9)
    return copyT if get_leaf_set_ete(copyT) else None

def selection_of_mcs(T_i, T_set, U, p=0.5, multi_leaf=True):
    C=set().union(*(extract_clusters(Tj,U,multi_leaf) for Tj in T_set))
    if not C: return []
    wt = compute_weights_globally()
    C_p,_ = frequency_and_filter(C,T_set,p,wt)
    if not C_p: return []
    G = max_coverage_group(group_clusters_by_leafset(C_p),U)
    if not G: return []

    S_list=[]
    for c in G:
        for idx,T_j in enumerate(T_set):
            if c.issubset(get_leaf_set_ete(T_j)):
                conn_len = 1.0
                if len(c)==1:
                    lf = next(iter(c))
                    nd = T_j&lf
                    conn_len = nd.dist if nd.dist>0 else 1.0
                else:
                    try:
                        mm = T_j.get_common_ancestor(list(c))
                        conn_len = mm.dist if mm and mm.dist>0 else 1.0
                    except: pass
                sub = extract_subtree(c,T_j)
                if sub: S_list.append((sub,T_j,idx,float(conn_len)))
    return S_list


def build_consensus_mcs(S_list, p=0.5):
    if not S_list: return [],[]
    max_leaf_count=max(len(get_leaf_set_ete(x[0])) for x in S_list)
    connecting_lens=[x[3] for x in S_list]
    combined_avg=(sum(connecting_lens)/len(connecting_lens)
                  if connecting_lens else 1.0)
    combined_avg=max(combined_avg,1e-9)

    # single-leaf consensus
 
    if max_leaf_count==1:
        leaf2lens=defaultdict(list)
        for (Sj,_,_,cl) in S_list:
            for lf in get_leaf_set_ete(Sj): leaf2lens[lf].append(cl)
        single_trees, host_lists = [], []
        for lf,lens in leaf2lens.items():
            t=Tree();  t.add_child(name=lf, dist=max(sum(lens)/len(lens),1e-9))
            t.connecting_length=t.children[0].dist
            single_trees.append(t)
            host_lists.append(sorted(
                {host_idx for (S,_,host_idx,_) in S_list
                 if lf in get_leaf_set_ete(S)}))
        return single_trees, host_lists

    # multi-leaf consensus
 
    consensus_tree, host_idxs = build_weighted_consensus_topology(S_list,p,global_T_i)
    if not consensus_tree or len(consensus_tree)==0: return [],[]
    consensus_tree.connecting_length = combined_avg
    wt = compute_weights_globally()
    avg_dist,_ = compute_average_distances(S_list, wt)
    if len(consensus_tree.get_leaves())>2:
        consensus_tree = least_squares_fit_branch_lengths(consensus_tree, avg_dist)
    elif len(consensus_tree.get_leaves())==2:
        a, b   = consensus_tree.get_leaves()
        la, lb = a.name, b.name
        cA, cB = [], []
        w_map  = compute_weights_globally()
        w_sum  = 0.0
        for (Sj, _, host_idx, _) in S_list:
            w = w_map.get(host_idx, 0.0)
            if w == 0.0:
                continue
            if {la, lb}.issubset(get_leaf_set_ete(Sj)):
                mm  = Sj.get_common_ancestor(la, lb)
                cA.append(w * node_to_ancestor_distance(Sj & la, mm))
                cB.append(w * node_to_ancestor_distance(Sj & lb, mm))
                w_sum += w

        if w_sum > 0:
            a.dist = max(sum(cA) / w_sum, 1e-9)
            b.dist = max(sum(cB) / w_sum, 1e-9)

    return [consensus_tree],[host_idxs]

# ────────────────────────────────────────────────────────────────
#  MAIN part
# ────────────────────────────────────────────────────────────────

def _ms_label(path):
    m = re.search(r"multiset_(\d+)\.txt$", os.path.basename(path))
    return int(m.group(1)) if m else path

def complete_all_multisets():
    in_dir  = "input_multisets"
    out_dir = "completed_multisets"
    os.makedirs(out_dir, exist_ok=True)
    
    # find all files  input_multisets/multiset_*.txt
    pattern    = os.path.join(in_dir, "multiset_*.txt")
    file_list  = sorted(glob.glob(pattern), key=_ms_label)
    
    for fin in file_list:
        ms_i  = _ms_label(fin)          # for log / output file name
        fout  = os.path.join(out_dir, f"completed_multiset_{ms_i}.txt")
        print(f"Processing multiset {ms_i} …")

        #  read trees
        with open(fin) as f:
            lines = [preprocess_newick(l.strip()) for l in f if l.strip()]
        all_trees=[]
        for line in lines:
            t=Tree(line,format=1)
            for n in t.traverse("postorder"):    # ensure positive lengths
                n.dist = max(n.dist or 0.0, 1e-9)
            all_trees.append(t)

        overall_leaves=set().union(*(get_leaf_set_ete(t) for t in all_trees))
        completed=[]

        for idx,orig in enumerate(all_trees,1):
            print(f"  completing tree {idx}/{len(all_trees)}")
            tgt_updated = orig.copy("deepcopy")
            host_list   = [all_trees[j].copy("deepcopy")
                           for j in range(len(all_trees)) if j!=idx-1]
            
            global global_T_set, global_T_i
            global_T_set = host_list
            global_T_i   = tgt_updated

            
            anchor_leaves_original = sorted(
                get_leaf_set_ete(orig) & get_common_leaves(host_list)
            )
            U = overall_leaves - get_leaf_set_ete(tgt_updated)
            inserted_leaves=set()

            p=0.5; p_reduced=False
            while U and p>=0:
                # multi-leaf MCS first
                from_old_script_selection_of_mcs = selection_of_mcs(
                    tgt_updated, host_list, U, p, multi_leaf=True)
                if from_old_script_selection_of_mcs:
                    c_subtrees, host_idxs = build_consensus_mcs(
                        from_old_script_selection_of_mcs, p)
                    if c_subtrees:
                        before=len(U)
                        for S_star, hlist in zip(c_subtrees, host_idxs):
                            hosts=[host_list[h] for h in hlist]
                            ok = insert_subtree_kncl(
                                tgt_updated, orig, S_star, hosts,
                                anchor_leaves_original,
                                K_USER
                            )
                            if ok:
                                inserted_leaves |= get_leaf_set_ete(S_star)
                                U -= get_leaf_set_ete(S_star)
                        if len(U)<before: continue

                # single-leaf MCS
                from_old_script_selection_single = selection_of_mcs(
                    tgt_updated, host_list, U, p, multi_leaf=False)
                if from_old_script_selection_single:
                    c_subtrees, host_idxs = build_consensus_mcs(
                        from_old_script_selection_single, p)
                    if c_subtrees:
                        before=len(U)
                        for S_star, hlist in zip(c_subtrees, host_idxs):
                            hosts=[host_list[h] for h in hlist]
                            ok = insert_subtree_kncl(
                                tgt_updated, orig, S_star, hosts,
                                anchor_leaves_original,
                                K_USER
                            )
                            if ok:
                                inserted_leaves |= get_leaf_set_ete(S_star)
                                U -= get_leaf_set_ete(S_star)
                        if len(U)<before: continue

                p_reduced=True; p-=0.1

            leftover = U & (overall_leaves-get_leaf_set_ete(tgt_updated))
            if leftover:
                print(f"   [WARN] still missing {len(leftover)} leaves {leftover}")
            else:
                print("   done.")

            completed.append(tgt_updated)

        # --- write output ---
        with open(fout,"w") as fo:
            for t in completed:
                clear_internal_node_names(t)
                fo.write(t.write(format=1)+"\n")
    print("All multisets done.")

# Main part
if __name__ == "__main__":
    complete_all_multisets()
