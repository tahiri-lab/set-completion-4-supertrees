#!/usr/bin/env python3

import os
import re
import math
import numpy as np
from collections import defaultdict
from itertools import combinations
from ete3 import Tree

##############################
# HELPER FUNCTIONS
##############################

def preprocess_newick(newick_str):
    """Replaces spaces with underscores in taxon labels within a Newick string."""
    return re.sub(r'(?<=\w) (?=\w)', '_', newick_str)

def get_leaf_set_ete(tree):
    """Returns a set of leaf names from an ETE tree."""
    return set(leaf.name for leaf in tree.get_leaves() if leaf.name)

def node_to_ancestor_distance(node, ancestor):
    """Distance from `node` up to `ancestor`."""
    dist = 0.0
    curr = node
    while curr and curr != ancestor:
        if curr.dist:
            dist += curr.dist
        curr = curr.up
    return dist

def is_rooted_ete(tree):
    """If the root node has <=2 children => treat as basically rooted."""
    root = tree
    while root.up:
        root = root.up
    return (len(root.children) <= 2)

def midpoint_root(tree):
    """If not recognized as rooted, try rooting."""
    try:
        og = tree.get_midpoint_outgroup()
        if og == tree:
            return
        tree.set_outgroup(og)
    except:
        pass

def least_squares_fit_branch_lengths(consensus_tree, avg_dist):
    """Adjust branch lengths in `consensus_tree` via least squares to fit `avg_dist`."""
    if not is_rooted_ete(consensus_tree):
        midpoint_root(consensus_tree)

    for node in consensus_tree.traverse("postorder"):
        if node.dist is None or node.dist <= 0:
            node.dist = 1e-9

    edges = []
    node_to_edge = {}
    idx = 0
    for nd in consensus_tree.traverse("postorder"):
        if nd.up is not None:
            edges.append((nd, nd.up))
            node_to_edge[nd] = idx
            idx += 1

    leaves = [lf.name for lf in consensus_tree.get_leaves() if lf.name]
    leaves.sort()
    leaf_map = {lf.name: lf for lf in consensus_tree.get_leaves() if lf.name}

    M = []
    d_vec = []
    for (l_a, l_b), dist_val in avg_dist.items():
        if l_a not in leaf_map or l_b not in leaf_map:
            continue
        node_a = leaf_map[l_a]
        node_b = leaf_map[l_b]

        path_a = []
        curr = node_a
        while curr and curr.up:
            path_a.append(curr)
            curr = curr.up
        path_b = []
        curr = node_b
        while curr and curr.up:
            path_b.append(curr)
            curr = curr.up

        # Remove common tail
        while path_a and path_b and path_a[-1] == path_b[-1]:
            path_a.pop()
            path_b.pop()

        row_vec = np.zeros(len(edges))
        for nd in path_a:
            if nd in node_to_edge:
                row_vec[node_to_edge[nd]] += 1.0
        for nd in path_b:
            if nd in node_to_edge:
                row_vec[node_to_edge[nd]] += 1.0

        M.append(row_vec)
        d_vec.append(dist_val)

    M = np.array(M)
    d_vec = np.array(d_vec)

    if M.size > 0 and M.shape[1] > 0:
        try:
            x, _, _, _ = np.linalg.lstsq(M, d_vec, rcond=None)
            for nd, eidx in node_to_edge.items():
                length = x[eidx]
                if length < 1e-9:
                    length = 1e-9
                nd.dist = length
        except np.linalg.LinAlgError:
            pass

    return consensus_tree

##############################
# Weighted majority splits
##############################

def extract_splits(tree):
    """Extract bipartition splits, taking the smaller side of each."""
    splits = []
    total_leaves = len(get_leaf_set_ete(tree))
    for nd in tree.traverse("postorder"):
        if nd.is_leaf() or nd.up is None:
            continue
        leaves_under = set(x.name for x in nd.get_leaves() if x.name)
        if len(leaves_under) <= total_leaves / 2:
            splits.append(frozenset(leaves_under))
    return splits

def compute_weights_globally():
    """Uses global_T_set and global_T_i to compute weights for each host tree."""
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

def build_weighted_consensus_topology(S_list, p, T_i):
    from collections import defaultdict
    all_leaves = set()
    root_clade_counts = defaultdict(float)
    split_counts = defaultdict(float)

    all_hosts = [h_idx for (_, _, h_idx, _) in S_list]
    w_j = compute_weights_globally()
    sum_w = sum(w_j.values())

    for (S_j, _, host_idx, _) in S_list:
        leaves_j = get_leaf_set_ete(S_j)
        all_leaves.update(leaves_j)
        r_j = frozenset(leaves_j)
        root_clade_counts[r_j] += w_j[host_idx]

        sp_list = extract_splits(S_j)
        for sp in sp_list:
            split_counts[sp] += w_j[host_idx]

    split_freq = {}
    for sp, wsum in split_counts.items():
        freq = wsum / sum_w if sum_w > 0 else 0
        split_freq[sp] = freq

    majority_splits = [sp for sp in split_freq if split_freq[sp] > p]

    root_freq = {}
    for rc, wsum in root_clade_counts.items():
        fv = wsum / sum_w if sum_w > 0 else 0
        root_freq[rc] = fv

    best_root = None
    best_val = -1
    for r, val in root_freq.items():
        if val > best_val:
            best_val = val
            best_root = r

    consensus_tree = build_consensus_topology_from_splits_and_root(all_leaves, majority_splits, best_root)
    return consensus_tree, sorted(set(all_hosts))

def build_consensus_topology_from_splits_and_root(leaves, majority_splits, best_root_clade):
    """Build from splits, forcibly root at best_root_clade."""
    t = Tree()
    t.name = "WeightedConsensus"
    t.dist = 0.0

    if not leaves:
        return t
    if len(leaves) == 1:
        lf = list(leaves)[0]
        t.add_child(name=lf, dist=1.0)
        return t

    for label in leaves:
        t.add_child(name=label, dist=1.0)

    majority_splits_sorted = sorted(majority_splits, key=lambda s: len(s), reverse=True)
    for sp in majority_splits_sorted:
        sp_list = list(sp)
        try:
            mrca = t.get_common_ancestor(sp_list)
            if mrca and mrca.up:
                mrca.dist = 1.0
        except:
            continue

    all_leaves = set(x.name for x in t.get_leaves() if x.name)
    if best_root_clade and 0 < len(best_root_clade) < len(all_leaves):
        outside = list(all_leaves - set(best_root_clade))
        if outside:
            out_label = outside[0]
            out_node = t.search_nodes(name=out_label)
            if out_node:
                out_node = out_node[0]
                try:
                    t.set_outgroup(out_node)
                except:
                    pass

    for nd in t.traverse("postorder"):
        if nd.up and (nd.dist is None or nd.dist <= 0):
            nd.dist = 1e-9
    return t

def build_consensus_mcs(S_list, p=0.5):
    """
    Build a single consensus subtree from S_list.
    We store the average "connecting length" in .connecting_length,
    set the root dist=0 => no double count if multi-leaf.
    """
    if not S_list:
        return [], []

    leaf_counts = [len(get_leaf_set_ete(x[0])) for x in S_list]
    max_leaf_count = max(leaf_counts)

    connecting_lengths = [x[3] for x in S_list]
    if connecting_lengths:
        combined_avg = sum(connecting_lengths) / len(connecting_lengths)
    else:
        combined_avg = 1.0
    if combined_avg <= 0:
        combined_avg = 1.0

    if max_leaf_count == 1:
        from collections import defaultdict
        leaf2lengths = defaultdict(list)
        for (S_j, _, host_idx, c_len) in S_list:
            leaves_j = get_leaf_set_ete(S_j)
            for lf_label in leaves_j:
                leaf2lengths[lf_label].append(c_len)

        single_leaf_trees = []
        single_leaf_hosts = []
        for lf_label, lengths in leaf2lengths.items():
            t = Tree()
            t.add_child(name=lf_label, dist=0.0)
            avg_len = sum(lengths)/len(lengths) if lengths else 1.0
            if avg_len <= 0:
                avg_len = 1.0
            leaf_node = t.search_nodes(name=lf_label)[0]
            leaf_node.dist = avg_len
            t.connecting_length = avg_len
            single_leaf_trees.append(t)

            relevant_hosts = []
            for (subS, _, h_idx, _) in S_list:
                if lf_label in get_leaf_set_ete(subS):
                    relevant_hosts.append(h_idx)
            relevant_hosts = sorted(set(relevant_hosts))
            single_leaf_hosts.append(relevant_hosts)

        return single_leaf_trees, single_leaf_hosts
    else:
        consensus_tree, host_idxs = build_weighted_consensus_topology(S_list, p, global_T_i)
        if not consensus_tree or len(consensus_tree.get_leaves()) == 0:
            return [], []

        consensus_tree.connecting_length = combined_avg
        consensus_tree.dist = 0.0  # avoid double count

        wt = compute_weights_globally()
        avg_dist, _ = compute_average_distances(S_list, wt)

        leaves_cons = consensus_tree.get_leaves()
        if len(leaves_cons) == 2:
            la, lb = leaves_cons[0].name, leaves_cons[1].name
            cA = []
            cB = []
            for (S_j, _, _, _) in S_list:
                lset = get_leaf_set_ete(S_j)
                if la in lset and lb in lset:
                    try:
                        mm = S_j.get_common_ancestor(la, lb)
                        if mm:
                            da = node_to_ancestor_distance(S_j & la, mm)
                            db = node_to_ancestor_distance(S_j & lb, mm)
                            cA.append(da)
                            cB.append(db)
                    except:
                        continue
            if cA and cB:
                Aavg = sum(cA)/len(cA)
                Bavg = sum(cB)/len(cB)
            else:
                Aavg = Bavg = 1.0
            leaves_cons[0].dist = max(Aavg, 1e-9)
            leaves_cons[1].dist = max(Bavg, 1e-9)
        else:
            if len(leaves_cons) > 2:
                consensus_tree = least_squares_fit_branch_lengths(consensus_tree, avg_dist)

        return [consensus_tree], [host_idxs]

def extract_clusters(tree, U, multi_leaf=True):
    clusters = []
    for nd in tree.traverse("postorder"):
        if nd.is_leaf():
            # Single leaf cluster (if multi_leaf=False)
            if not multi_leaf:
                if nd.name and nd.name in U:
                    clusters.append(frozenset([nd.name]))
            continue
        leaves_under = {x.name for x in nd.get_leaves() if x.name}
        if leaves_under and leaves_under.issubset(U):
            if multi_leaf and len(leaves_under) > 1:
                clusters.append(frozenset(leaves_under))
            elif not multi_leaf and len(leaves_under) == 1:
                clusters.append(frozenset(leaves_under))
    return clusters

def frequency_and_filter(C, T_set, p, wt):
    sum_w = sum(wt[idx] for idx in range(len(T_set)))
    if sum_w <= 0:
        return [], {}

    C_freq = {}
    for c in C:
        wsum = 0.0
        for idx, T_j in enumerate(T_set):
            if c.issubset(get_leaf_set_ete(T_j)):
                wsum += wt.get(idx, 0.0)
        freq = wsum/sum_w
        C_freq[c] = freq
    C_p = [x for x in C if C_freq[x] > p]
    return C_p, C_freq

def group_clusters_by_leafset(C_p):
    from collections import defaultdict
    groups = defaultdict(list)
    for c in C_p:
        groups[frozenset(c)].append(c)
    return list(groups.values())

def max_coverage_group(groups, U):
    best = None
    best_cov = -1
    for G in groups:
        leaf_set = set()
        for sp in G:
            leaf_set.update(sp)
        coverage = len(leaf_set & U)
        if coverage > best_cov:
            best_cov = coverage
            best = G
    return best

def extract_subtree(C, T):
    if not C:
        return None
    copyT = T.copy(method='deepcopy')
    all_leaves_in_copy = {x.name for x in copyT.get_leaves() if x.name}
    leaves_to_prune = all_leaves_in_copy - C
    for lf in leaves_to_prune:
        node_list = copyT.search_nodes(name=lf)
        for nd in node_list:
            nd.detach()
    for nd in copyT.traverse("postorder"):
        if nd.dist is None or nd.dist <= 0:
            nd.dist = 1e-9
    rema = get_leaf_set_ete(copyT)
    if not rema:
        return None
    return copyT

def selection_of_mcs(T_i, T_set, U, p=0.5, multi_leaf=True):
    C = []
    for T_j in T_set:
        c_j = extract_clusters(T_j, U, multi_leaf=multi_leaf)
        C.extend(c_j)
    C = list(set(C))
    if not C:
        return []
    wt = compute_weights_globally()
    C_p, _ = frequency_and_filter(C, T_set, p, wt)
    if not C_p:
        return []
    groups = group_clusters_by_leafset(C_p)
    G_max = max_coverage_group(groups, U)
    if not G_max:
        return []

    S_list = []
    for c in G_max:
        for idx, T_j in enumerate(T_set):
            if c.issubset(get_leaf_set_ete(T_j)):
                if len(c) == 1:
                    leaf_label = next(iter(c))
                    lf_nodes = T_j.search_nodes(name=leaf_label)
                    if lf_nodes and lf_nodes[0].dist > 0:
                        conn_len = lf_nodes[0].dist
                    else:
                        conn_len = 1.0
                else:
                    try:
                        mm = T_j.get_common_ancestor(list(c))
                        conn_len = mm.dist if (mm and mm.dist > 0) else 1.0
                    except:
                        conn_len = 1.0
                sub_copy = extract_subtree(c, T_j)
                if sub_copy:
                    S_list.append((sub_copy, T_j, idx, float(conn_len)))
    return S_list

##############################
# Global references
##############################

TEMP_PREFIX = "__temp__"

global_T_set = []
global_T_i = None

def compute_average_distances(S_list, wt):
    alls = set()
    for (S_j, _, _, _) in S_list:
        alls |= get_leaf_set_ete(S_j)
    alls = sorted(alls)

    dist_sum = {}
    for i in range(len(alls)):
        for j in range(i+1, len(alls)):
            dist_sum[(alls[i], alls[j])] = 0.0

    sum_w = sum(wt.values())
    for (S_j, _, host_idx, _) in S_list:
        w_j = wt.get(host_idx, 0)
        lset = get_leaf_set_ete(S_j)
        for i in range(len(alls)):
            for j in range(i+1, len(alls)):
                la = alls[i]
                lb = alls[j]
                if la in lset and lb in lset:
                    try:
                        mm = S_j.get_common_ancestor(la, lb)
                    except:
                        continue
                    nd_a = S_j.search_nodes(name=la)
                    nd_b = S_j.search_nodes(name=lb)
                    if nd_a and nd_b and mm:
                        da = node_to_ancestor_distance(nd_a[0], mm)
                        db = node_to_ancestor_distance(nd_b[0], mm)
                        d_ij = da + db
                    else:
                        d_ij = 0.0
                    dist_sum[(la, lb)] += w_j * d_ij

    avg_dist = {}
    for (la, lb), val in dist_sum.items():
        if sum_w > 0:
            avg_dist[(la, lb)] = val/sum_w
        else:
            avg_dist[(la, lb)] = 0.0
    return avg_dist, alls

##############################
# Naming internal nodes
##############################

def assign_internal_node_names(tree, prefix="INODE"):
    """
    Assigns unique names to internal nodes so that they can be tracked more easily.
    """
    idx = 1
    for nd in tree.traverse("preorder"):
        if not nd.is_leaf():
            nd.name = f"{prefix}_{idx}"
            idx += 1
            
def clear_internal_node_names(tree):
    """
    Clears the names of all internal nodes in the tree.
    """
    for node in tree.traverse():
        if not node.is_leaf():
            node.name = ""

##############################
# Insert BFS
##############################

def InsertTempLeaves(tree,
                     target_leaf,
                     new_leaf_base_name,
                     new_length,
                     dist,
                     inserted_leaves_global,
                     inserted_subtree_leaves_global=None,
                     tolerance=1e-10):
    """
    BFS with stable sorting => consistent expansions
    using two sets:
      - inserted_leaves_global: placeholders
      - inserted_subtree_leaves_global: real leaves from previously inserted subtrees
    Also uses TEMP_PREFIX for placeholders.
    """
    if inserted_subtree_leaves_global is None:
        inserted_subtree_leaves_global = set()

    insertion_points = []
    visited_nodes = set()
    target_node_list = tree.search_nodes(name=target_leaf)
    if not target_node_list:
        return insertion_points
    target_node = target_node_list[0]

    def robust_insert_leaf_at_node(current_node,
                                   insert_distance,
                                   previous_node,
                                   original_branch_distance,
                                   toward_root=False):
        excess_length = original_branch_distance - insert_distance
        if excess_length < 0:
            excess_length = 0
        if toward_root:
            temp = current_node
            current_node = previous_node
            previous_node = temp

        parent = previous_node.up
        if parent:
            previous_node.detach()
        else:
            parent = tree

        new_leaf_name = f"{TEMP_PREFIX}{target_leaf}_{new_leaf_base_name}{len(insertion_points)+1}"
        new_internal_node = parent.add_child(dist=excess_length)
        new_internal_node.add_child(previous_node, dist=insert_distance)
        new_internal_node.add_child(name=new_leaf_name, dist=new_length)

        insertion_points.append(new_leaf_name)
        visited_nodes.add(new_internal_node)
        inserted_leaves_global.add(new_leaf_name)
        return True

    def insert_leaf_at_terminal(current_node, insert_distance):
        if (current_node.name in inserted_leaves_global or
            (current_node.name and current_node.name.startswith(TEMP_PREFIX)) or
            (current_node.name in inserted_subtree_leaves_global)):
            return False

        excess_length = current_node.dist - insert_distance
        if excess_length < 0:
            excess_length = 0
        parent = current_node.up
        if parent:
            current_node.detach()
            new_internal_node = parent.add_child(dist=excess_length)
            new_internal_node.add_child(current_node, dist=insert_distance)

            new_leaf_name = f"{TEMP_PREFIX}{target_leaf}_{new_leaf_base_name}{len(insertion_points)+1}"
            new_internal_node.add_child(name=new_leaf_name, dist=new_length)

            insertion_points.append(new_leaf_name)
            visited_nodes.add(new_internal_node)
            inserted_leaves_global.add(new_leaf_name)
        else:
            return False
        return True

    def bfs(start_node, accumulated_dist):
        queue = [(start_node, accumulated_dist, None, 0, [], False)]
        while queue:
            (curr_node, curr_dist, prev_node, prev_dist, path, toward_root) = queue.pop(0)

            # Skip expansions if visited or if it's a placeholder or from an inserted subtree
            if (curr_node in visited_nodes or
                (curr_node.name and curr_node.name.startswith(TEMP_PREFIX)) or
                (curr_node.name in inserted_leaves_global) or
                (curr_node.name in inserted_subtree_leaves_global)):
                continue

            visited_nodes.add(curr_node)

            if round(curr_dist, 8) >= dist:
                insert_distance = round(curr_dist, 8) - round(dist, 8)
                if abs(insert_distance) < tolerance:
                    insert_distance = 0
                if insert_distance == 0:
                    robust_insert_leaf_at_node(curr_node, insert_distance, prev_node, curr_node.dist, toward_root)
                elif curr_node.is_leaf():
                    insert_leaf_at_terminal(curr_node, insert_distance)
                else:
                    robust_insert_leaf_at_node(prev_node,
                                               prev_dist - insert_distance,
                                               curr_node,
                                               prev_dist,
                                               toward_root)
                continue

            # stable sort children
            children_sorted = sorted(curr_node.children, key=lambda x: (x.name or ''))
            for child in children_sorted:
                if (child not in visited_nodes and
                    (not child.name or not child.name.startswith(TEMP_PREFIX)) and
                    (child.name not in inserted_leaves_global) and
                    (child.name not in inserted_subtree_leaves_global)):
                    queue.append((child,
                                  curr_dist + child.dist,
                                  curr_node,
                                  child.dist,
                                  path+[curr_node.name],
                                  False))

            # stable parent insertion
            if (curr_node.up and
                (curr_node.up not in visited_nodes) and
                (not curr_node.up.name or not curr_node.up.name.startswith(TEMP_PREFIX)) and
                (curr_node.up.name not in inserted_leaves_global) and
                (curr_node.up.name not in inserted_subtree_leaves_global)):
                queue.append((curr_node.up,
                              curr_dist + curr_node.dist,
                              curr_node,
                              curr_node.dist,
                              path+[curr_node.name],
                              True))

    if dist <= target_node.dist:
        insert_leaf_at_terminal(target_node, dist)
    else:
        bfs(target_node, 0)

    return insertion_points

def find_farthest_leaf(tree, start, temporary_leaves):
    placeholders_sorted = sorted(temporary_leaves)
    max_distance = 0
    farthest_leaf = start
    for lf_name in placeholders_sorted:
        if lf_name != start.name:
            lf = tree & lf_name
            d = tree.get_distance(start, lf)
            if d > max_distance:
                max_distance = d
                farthest_leaf = lf
    return farthest_leaf, max_distance

def find_path(leaf1, leaf2):
    path1 = []
    node = leaf1
    while node:
        path1.append(node)
        node = node.up
    path2 = []
    node = leaf2
    while node:
        path2.append(node)
        node = node.up
    lca = None
    for n1 in path1:
        if n1 in path2:
            lca = n1
            break

    path = []
    for nd in path1:
        path.append(nd)
        if nd == lca:
            break

    lca_index = path2.index(lca)
    reversed_path2 = list(reversed(path2[:lca_index]))
    path.extend(reversed_path2)

    branch_lengths = []
    for i in range(1, len(path)):
        branch_lengths.append(path[i-1].get_distance(path[i]))
    return path, branch_lengths

def compute_midpoint(tree, temporary_leaves):
    """
    For multi placeholder logic: find the midpoint among the placeholders
    and return (prev_node, curr_node, excess, half_dist, last_branch_dist).
    """
    placeholders_sorted = sorted(temporary_leaves)
    if not placeholders_sorted:
        return None, None, 0, 0, 0

    start_name = placeholders_sorted[0]
    start = tree & start_name
    leaf1, _ = find_farthest_leaf(tree, start, placeholders_sorted)
    leaf2, max_diam = find_farthest_leaf(tree, leaf1, placeholders_sorted)
    path, branch_lengths = find_path(leaf1, leaf2)
    total_distance = max_diam
    half_distance = round(total_distance/2, 8)

    cumulative_distance = 0
    prev_node = None
    for i,node in enumerate(path):
        if i>0:
            cumulative_distance += round(branch_lengths[i-1],8)
        if cumulative_distance >= half_distance:
            excess = round(cumulative_distance - half_distance,8)
            prev_node = path[i-1]
            return prev_node, node, excess, half_distance, branch_lengths[i-1] if i>0 else 0
        elif abs(cumulative_distance - half_distance) < 1e-10:
            prev_node = path[i]
            return prev_node, path[i], 0, half_distance, branch_lengths[i-1] if i>0 else 0

    return None, None, 0, 0, 0

def insert_midpoint_and_new_subtree(tree, prev_node, curr_node, excess, subtree,
                                    branch_length, original_dist):
    if not curr_node or not prev_node:
        return tree

    distance_to_midpoint = round(excess,8)
    distance_from_midpoint_to_leaf = round(original_dist - distance_to_midpoint,8)

    distance_to_midpoint = max(distance_to_midpoint,1e-9)
    distance_from_midpoint_to_leaf = max(distance_from_midpoint_to_leaf,1e-9)

    if curr_node.up == prev_node:
        parent = prev_node
        child  = curr_node
    elif prev_node.up == curr_node:
        parent = curr_node
        child  = prev_node
    else:
        return tree

    if child not in parent.children:
        return tree

    parent.remove_child(child)
    new_node = parent.add_child(dist=distance_to_midpoint)
    new_node.add_child(child, dist=distance_from_midpoint_to_leaf)

    new_subtree = subtree.copy(method='deepcopy')
    new_subtree.dist = max(branch_length,1e-9)
    new_node.add_child(new_subtree)

    return tree

##############################
# FALLBACK for entire subtree
##############################

def fallback_insert_subtree_entire(T_i_updated,
                                   S_star_adj,
                                   tau_global):
    """
    Remove partial placeholders if any, then attach S_star_adj at midpoint or root.
    """
    # remove partial placeholders
    partial_temp = [x for x in T_i_updated.get_leaf_names() if x.startswith(TEMP_PREFIX)]
    if partial_temp:
        #print(f"[FALLBACK NOTICE] Removing {len(partial_temp)} partial placeholder(s) before fallback.")
        keep = [x for x in T_i_updated.get_leaf_names() if not x.startswith(TEMP_PREFIX)]
        T_i_updated.prune(keep, preserve_branch_length=True)

    if hasattr(S_star_adj,'connecting_length') and S_star_adj.connecting_length>0:
        attach_len = S_star_adj.connecting_length*tau_global
    else:
        attach_len = 1.0

    for nd in S_star_adj.traverse():
        nd.dist *= tau_global

    T_leaves = T_i_updated.get_leaves()
    if len(T_leaves) < 2:
        #print("[FALLBACK NOTICE] T_i has <2 leaves => fallback attach at root.")
        T_i_updated.add_child(S_star_adj, dist=attach_len)
    else:
        placeholders_main = [lf.name for lf in T_leaves]
        prev_node, curr_node, excess, _, orig_dist = compute_midpoint(T_i_updated, placeholders_main)
        if (not prev_node) or (not curr_node):
            #print("[FALLBACK NOTICE] Could not compute midpoint => fallback attach at root.")
            T_i_updated.add_child(S_star_adj, dist=attach_len)
        else:
            #print("[FALLBACK NOTICE] Attaching entire subtree via fallback.")
            T_i_updated = insert_midpoint_and_new_subtree(
                T_i_updated, prev_node, curr_node, excess,
                S_star_adj, attach_len, orig_dist
            )

    assign_internal_node_names(T_i_updated)
    return T_i_updated

##############################
# Insertions
##############################

class AmbiguousNodeError(Exception):
    """Custom exception for ambiguous node name errors."""
    pass

def compute_global_adjustment_rate(T_i_original, host_trees, anchor_leaves):
    """
    Single global scale factor tau,
    using the original T_i for distances among anchors,
    but the actual host trees for this insertion.
    """
    if not anchor_leaves or not host_trees:
        return 1.0

    anchor_list = list(anchor_leaves)
    pairwise = []
    for i in range(len(anchor_list)):
        for j in range(i+1, len(anchor_list)):
            la = anchor_list[i]
            lb = anchor_list[j]
            # Skip if either anchor is missing in original T_i
            if not T_i_original.search_nodes(name=la) or not T_i_original.search_nodes(name=lb):
                continue
            dval = T_i_original.get_distance(la, lb)
            pairwise.append((la, lb, dval))

    if not pairwise:
        return 1.0

    sum_Ti = sum(x[2] for x in pairwise)
    n_pairs = len(pairwise)
    avg_Ti = sum_Ti / n_pairs if n_pairs else 0

    # We now measure how host trees see these same anchor pairs
    sum_hosts = 0.0
    count_hosts = 0
    for Th in host_trees:
        partial_sum = 0.0
        partial_count = 0
        for (la, lb, _) in pairwise:
            if Th.search_nodes(name=la) and Th.search_nodes(name=lb):
                partial_sum += Th.get_distance(la, lb)
                partial_count += 1
        if partial_count > 0:
            sum_hosts += (partial_sum / partial_count)
            count_hosts += 1

    if count_hosts == 0:
        return 1.0

    avg_hosts = sum_hosts / count_hosts
    if avg_hosts > 0:
        tau = avg_Ti / avg_hosts
    else:
        tau = 1.0
    return tau

def compute_leaf_lca_distance_from_hosts(c, host_trees, subtree_leaves):
    """
    Among the given host trees, compute the average distance from c to the MRCA of subtree_leaves.
    """
    sum_d = 0.0
    count = 0
    needed_leaves = list(subtree_leaves)
    for Th in host_trees:
        al = set(Th.get_leaf_names())
        if c not in al:
            continue
        if not set(needed_leaves).issubset(al):
            continue
        try:
            mrca = Th.get_common_ancestor(needed_leaves)
            dist_val = Th.get_distance(c, mrca)
            sum_d += dist_val
            count += 1
        except:
            continue
    if count > 0:
        return sum_d / count
    return 0.0

def compute_local_leaf_based_rate(T_i_original, host_trees, c, anchor_leaves):
    """
    For each anchor c, compare T_i_original distances to the actual host trees' distances.
    """
    others = anchor_leaves - {c}
    if not others or not host_trees:
        return 1.0

    # Distances in original T_i
    sumTi = 0.0
    countTi = 0
    for la in others:
        if T_i_original.search_nodes(name=c) and T_i_original.search_nodes(name=la):
            sumTi += T_i_original.get_distance(c, la)
            countTi += 1

    if countTi == 0:
        return 1.0
    avgTi = sumTi / countTi

    # Distances in the actual host trees
    sumHosts = 0.0
    countHosts = 0
    for Th in host_trees:
        if Th.search_nodes(name=c):
            partial = 0.0
            local_count = 0
            for la in others:
                if Th.search_nodes(name=la):
                    partial += Th.get_distance(c, la)
                    local_count += 1
            if local_count > 0:
                sumHosts += (partial / local_count)
                countHosts += 1

    if countHosts == 0:
        return 1.0

    avgHostsLeaf = sumHosts / countHosts
    if avgHostsLeaf <= 0:
        return 1.0
    ratio = avgTi / avgHostsLeaf
    return ratio

def BFS_subtree_insertion(T_i_updated,
                          T_i_original,
                          S_star_adj,
                          host_trees,
                          anchor_leaves_original,
                          inserted_subtree_leaves_global):
    """
    Perform BFS insertion on T_i_updated using anchor distances from T_i_original,
    but the actual host_trees for this insertion.
    """
    subtree_leaves = get_leaf_set_ete(S_star_adj)
    try:
        mrca_subtree = S_star_adj.get_common_ancestor(list(subtree_leaves))
    except:
        mrca_subtree = S_star_adj

    tau_global = compute_global_adjustment_rate(T_i_original, host_trees, set(anchor_leaves_original))

    inserted_placeholders_global = set()

    for c in anchor_leaves_original:
        # If c doesn't exist in T_i_updated, skip
        if not T_i_updated.search_nodes(name=c):
            continue

        local_rate = compute_local_leaf_based_rate(T_i_original, host_trees, c, set(anchor_leaves_original))

        if c in subtree_leaves:
            try:
                dist_c_mrca = S_star_adj.get_distance(c, mrca_subtree)
            except:
                dist_c_mrca = 0.0
        else:
            dist_c_mrca = compute_leaf_lca_distance_from_hosts(c, host_trees, subtree_leaves)

        dist_to_insert = dist_c_mrca * local_rate

        # Insert placeholders in the updated tree
        InsertTempLeaves(
            tree=T_i_updated,
            target_leaf=c,
            new_leaf_base_name="anchor",
            new_length=1.0,
            dist=dist_to_insert,
            inserted_leaves_global=inserted_placeholders_global,
            inserted_subtree_leaves_global=inserted_subtree_leaves_global
        )

    return inserted_placeholders_global, tau_global

def finalize_subtree_insertion(T_i_updated,
                               S_star_adj,
                               placeholders_global,
                               tau_global,
                               inserted_subtree_leaves_global):
    """
    After BFS, attach subtree using placeholders in T_i_updated.
    Then verify that subtree was actually attached; if not, fallback.
    """
    subtree_leaves_set = get_leaf_set_ete(S_star_adj)

    # Scale the subtree by tau_global
    if hasattr(S_star_adj, 'connecting_length') and S_star_adj.connecting_length > 0:
        attach_len = S_star_adj.connecting_length * tau_global
    else:
        attach_len = 1.0
    for nd in S_star_adj.traverse():
        nd.dist *= tau_global

    valid_placeholders = [x for x in placeholders_global if T_i_updated.search_nodes(name=x)]
    n_placeholders = len(valid_placeholders)

    # Try inserting the subtree via midpoint or direct attachment
    if n_placeholders == 0:
        T_i_updated.add_child(S_star_adj, dist=attach_len)

    elif n_placeholders == 1:
        single_name = valid_placeholders[0]
        node_list = T_i_updated.search_nodes(name=single_name)
        if node_list:
            nd = node_list[0]
            par = nd.up
            if par:
                br = max(nd.dist, 1e-9)
                par.remove_child(nd)
                par.add_child(S_star_adj, dist=br)

    elif n_placeholders == 2:
        lf1, lf2 = sorted(valid_placeholders)
        prev_node, curr_node, excess, _, br_len = compute_midpoint(T_i_updated, [lf1, lf2])
        if prev_node and curr_node:
            insert_midpoint_and_new_subtree(
                T_i_updated, prev_node, curr_node, excess,
                S_star_adj, attach_len, br_len
            )
    else:
        prev_node, curr_node, excess, _, br_len = compute_midpoint(T_i_updated, sorted(valid_placeholders))
        if prev_node and curr_node:
            insert_midpoint_and_new_subtree(
                T_i_updated, prev_node, curr_node, excess,
                S_star_adj, attach_len, br_len
            )

    # Now remove placeholders
    final_temp = [x for x in T_i_updated.get_leaf_names() if x.startswith(TEMP_PREFIX)]
    if final_temp:
        keep = [x for x in T_i_updated.get_leaf_names() if not x.startswith(TEMP_PREFIX)]
        T_i_updated.prune(keep, preserve_branch_length=True)

    # Assign internal node names again
    assign_internal_node_names(T_i_updated)

    # Final check: did we actually attach the subtree's leaves?
    new_leaves_in_tree = get_leaf_set_ete(T_i_updated)
    missing_still = subtree_leaves_set - new_leaves_in_tree
    if missing_still:
        # Fallback if something went wrong
        fallback_insert_subtree_entire(T_i_updated, S_star_adj, tau_global)
    else:
        # Mark that we've indeed inserted these leaves
        inserted_subtree_leaves_global.update(subtree_leaves_set)

    return True

def try_inserting_subtree(T_i_updated,
                          T_i_original,
                          S_star,
                          host_idx_list,
                          inserted_subtree_leaves_global,
                          anchor_leaves_original):
    """
    High-level function that tries BFS insertion for S_star.
    Rate computations use T_i_original plus the actual host trees (chosen by host_idx_list).
    The actual modifications happen in T_i_updated.
    """
    if not host_idx_list:
        return False

    # The "actual host trees" for this insertion
    host_trees = [global_T_set[h] for h in host_idx_list]

    # If no anchors exist, attach subtree at root
    if not anchor_leaves_original:
        print("[INFO] No anchor leaves => attach subtree at root by fallback.")
        fallback_insert_subtree_entire(T_i_updated, S_star, 1.0)
        inserted_subtree_leaves_global.update(get_leaf_set_ete(S_star))
        return True

    # Make a fresh copy of the subtree for scaling
    S_star_adj = S_star.copy(method='deepcopy')

    try:
        placeholders_global, tau_global = BFS_subtree_insertion(
            T_i_updated,
            T_i_original,
            S_star_adj,
            host_trees,
            anchor_leaves_original,
            inserted_subtree_leaves_global
        )
        finalize_subtree_insertion(
            T_i_updated,
            S_star_adj,
            placeholders_global,
            tau_global,
            inserted_subtree_leaves_global
        )
        return True

    except Exception as e:
        if "Ambiguous node name" in str(e):
            print("[FALLBACK NOTICE] Subtree had ambiguous node => restarting tree completion (fallback).")
            raise AmbiguousNodeError("Ambiguous node name encountered during subtree insertion.")
        else:
            raise

##############################
# MAIN
##############################

if __name__ == "__main__":
    input_folder = "input_multisets"
    output_folder = "completed_multisets"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    def get_common_leaves(trees):
        if not trees:
            return set()
        cc = get_leaf_set_ete(trees[0])
        for th in trees[1:]:
            cc &= get_leaf_set_ete(th)
        return cc

    for ms_i in range(1, 2):
        input_filename = os.path.join(input_folder, f"multiset_{ms_i}.txt")
        output_filename = os.path.join(output_folder, f"completed_multiset_{ms_i}.txt")
        print(f"Processing multiset {ms_i} ...")

        try:
            with open(input_filename, "r") as f:
                lines = [preprocess_newick(l.strip()) for l in f if l.strip()]
        except Exception as e:
            print(f"  [ERROR] Failed to read {input_filename}: {e}")
            continue

        all_trees = []
        for line_num, line in enumerate(lines, start=1):
            try:
                t_ete = Tree(line, format=1)
                for nd in t_ete.traverse("postorder"):
                    if nd.dist is None or nd.dist <= 0:
                        nd.dist = 1e-9
                all_trees.append(t_ete)
            except Exception as e:
                print(f"  [ERROR] Failed to parse tree at line {line_num} in multiset {ms_i}: {e}")
                continue

        # The union of all leaves in the multiset
        overall_leaves = set()
        for T_x in all_trees:
            overall_leaves |= get_leaf_set_ete(T_x)

        completed_list = []
        for i, original_tree in enumerate(all_trees, start=1):
            print(f"  Completing tree {i} of {len(all_trees)} in multiset {ms_i}")
            max_retries = 3
            retry_count = 0

            while retry_count < max_retries:
                try:
                    # Copy the target tree for updating
                    T_i_updated = original_tree.copy(method='deepcopy')
                    global global_T_i
                    global_T_i = T_i_updated

                    # Prepare the "host trees" for this iteration
                    host_list = []
                    for j, T_j in enumerate(all_trees):
                        if j != (i - 1):
                            host_list.append(T_j.copy(method='deepcopy'))

                    global global_T_set
                    global_T_set = host_list

                    # The set of leaves in the *original* T_i (no inserted leaves)
                    L_i_original = get_leaf_set_ete(original_tree)

                    # Intersection of all host trees' leaves
                    Lh_actual = get_common_leaves(host_list)

                    # Anchors from the original T_i and the actual host leaves
                    anchor_leaves_original = sorted(L_i_original & Lh_actual)

                    # The set of leaves not in T_i yet (that we must insert)
                    L_Ti = get_leaf_set_ete(T_i_updated)
                    U = overall_leaves - L_Ti

                    inserted_subtree_leaves_global = set()
                    p = 0.5
                    p_has_reduced = False

                    while U and p >= 0:
                        # Try multi-leaf clusters first
                        if not p_has_reduced:
                            S_list_multi = selection_of_mcs(T_i_updated, host_list, U, p, multi_leaf=True)
                            if S_list_multi:
                                ctrees_multi, hosts_multi = build_consensus_mcs(S_list_multi, p)
                                if ctrees_multi:
                                    new_size_before = len(U)
                                    for ctree, hlist in zip(ctrees_multi, hosts_multi):
                                        inserted = try_inserting_subtree(
                                            T_i_updated,
                                            original_tree,
                                            ctree,
                                            hlist,
                                            inserted_subtree_leaves_global,
                                            anchor_leaves_original
                                        )
                                        if inserted:
                                            U -= get_leaf_set_ete(ctree)
                                    new_size_after = len(U)
                                    if new_size_after < new_size_before:
                                        continue

                        # Next, try single-leaf clusters
                        S_list_single = selection_of_mcs(T_i_updated, host_list, U, p, multi_leaf=False)
                        if S_list_single:
                            ctrees_single, hosts_single = build_consensus_mcs(S_list_single, p)
                            if ctrees_single:
                                new_size_before = len(U)
                                for cct, hli in zip(ctrees_single, hosts_single):
                                    inserted = try_inserting_subtree(
                                        T_i_updated,
                                        original_tree,
                                        cct,
                                        hli,
                                        inserted_subtree_leaves_global,
                                        anchor_leaves_original
                                    )
                                    if inserted:
                                        U -= get_leaf_set_ete(cct)
                                new_size_after = len(U)
                                if new_size_after < new_size_before:
                                    continue

                        # If we reach here, reduce p and try again
                        p_has_reduced = True
                        p -= 0.1

                    leftover = U & (overall_leaves - get_leaf_set_ete(T_i_updated))
                    if leftover:
                        print(f"[WARNING] After completion, still missing {len(leftover)} leaves: {leftover}")
                    else:
                        print("[INFO] Tree completed successfully; all required leaves were inserted.")

                    completed_list.append(T_i_updated)
                    break  # Successfully completed the tree, exit retry loop

                except AmbiguousNodeError as e:
                    retry_count += 1
                    print(f"    [RETRY] Ambiguous node encountered. Retrying tree {i} (Attempt {retry_count}/{max_retries})")
                    if retry_count == max_retries:
                        print(f"    [ERROR] Failed to complete tree {i} after {max_retries} attempts due to ambiguous nodes.")
                except Exception as e:
                    print(f"    [ERROR] Unexpected error while processing tree {i} in multiset {ms_i}: {e}")
                    break  # Exit the retry loop for non-ambiguous errors

            if retry_count == max_retries:
                # Optionally handle or skip the failed tree
                continue

        # Write out all completed trees
        try:
            with open(output_filename, "w") as fout:
                for comp_tree in completed_list:
                    clear_internal_node_names(comp_tree)
                    fout.write(comp_tree.write(format=1) + "\n")
        except Exception as e:
            print(f"  [ERROR] Failed to write to {output_filename}: {e}")
            continue

    print("Done.")
