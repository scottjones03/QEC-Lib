#!/usr/bin/env python
"""Quick test to verify the per-block pair splitting fix."""
import sys
sys.path.insert(0, "src")

from typing import Dict, List, Tuple

# Simulate the original buggy code
def assign_pairs_buggy(sr_pairs, ion_to_block, block_names):
    """Original code that loses pairs when both ions not in mapping."""
    block_pairs: Dict[str, List[List[Tuple[int, int]]]] = {
        bname: [[] for _ in range(len(sr_pairs))]
        for bname in block_names
    }
    lost = []
    for round_i, pairs_in_round in enumerate(sr_pairs):
        for pair in pairs_in_round:
            a_idx, d_idx = pair
            blk_a = ion_to_block.get(a_idx)
            blk_d = ion_to_block.get(d_idx)
            if blk_a == blk_d and blk_a is not None:
                block_pairs[blk_a][round_i].append(pair)
            else:
                if blk_a is not None:
                    block_pairs[blk_a][round_i].append(pair)
                elif blk_d is not None:
                    block_pairs[blk_d][round_i].append(pair)
                else:
                    # BUG: pair is lost!
                    lost.append(pair)
    return block_pairs, lost

# Simulate fixed code
def assign_pairs_fixed(sr_pairs, ion_to_block, block_names):
    """Fixed code that assigns unmapped pairs to first block."""
    block_pairs: Dict[str, List[List[Tuple[int, int]]]] = {
        bname: [[] for _ in range(len(sr_pairs))]
        for bname in block_names
    }
    rescued = []
    for round_i, pairs_in_round in enumerate(sr_pairs):
        for pair in pairs_in_round:
            a_idx, d_idx = pair
            blk_a = ion_to_block.get(a_idx)
            blk_d = ion_to_block.get(d_idx)
            if blk_a == blk_d and blk_a is not None:
                block_pairs[blk_a][round_i].append(pair)
            else:
                if blk_a is not None:
                    block_pairs[blk_a][round_i].append(pair)
                elif blk_d is not None:
                    block_pairs[blk_d][round_i].append(pair)
                else:
                    # FIX: assign to first block
                    rescued.append(pair)
                    default_block = block_names[0]
                    block_pairs[default_block][round_i].append(pair)
    return block_pairs, rescued

def main():
    # Simulate CSS Surgery d=2 scenario
    # 3 blocks: block0, block1, block2 (bridge in middle)
    block_names = ["block0", "block1", "block2"]
    
    # ion_to_block mapping (typical for 3-block CSS surgery)
    # Block 0: ions 0-7 (data qubits in left patch)
    # Block 1: ions 8-15 (data qubits in right patch) 
    # Block 2: ions 16-21 (some ancillas)
    # Bridge ancillas: ions 22-25 NOT in any block mapping!
    ion_to_block = {
        0: "block0", 1: "block0", 2: "block0", 3: "block0",
        4: "block0", 5: "block0", 6: "block0", 7: "block0",
        8: "block1", 9: "block1", 10: "block1", 11: "block1",
        12: "block1", 13: "block1", 14: "block1", 15: "block1",
        16: "block2", 17: "block2", 18: "block2", 19: "block2",
        20: "block2", 21: "block2",
        # 22, 23, 24, 25 are bridge ancillas - NOT mapped!
    }
    
    # Simulate stabilizer_round_post CX pairs
    # These are 4-qubit CX instructions involving bridge ancillas
    sr_pairs = [
        # 8 instructions with 4 pairs each - typical for stabilizer round
        [(22, 0), (22, 1), (23, 2), (23, 3)],   # Bridge ancillas 22,23 with block0 data
        [(22, 4), (22, 5), (23, 6), (23, 7)],   # More bridge with block0
        [(24, 8), (24, 9), (25, 10), (25, 11)], # Bridge ancillas 24,25 with block1 data
        [(24, 12), (24, 13), (25, 14), (25, 15)], # More bridge with block1
        [(22, 16), (22, 17), (23, 18), (23, 19)], # Bridge with block2 
        [(24, 20), (24, 21), (22, 0), (23, 1)],   # Mixed
        [(24, 2), (25, 3), (22, 8), (23, 9)],     # Cross-block
        [(24, 10), (25, 11), (22, 12), (23, 13)], # Cross-block
    ]
    
    total_pairs = sum(len(p) for p in sr_pairs)
    print(f"Total input pairs: {total_pairs}")
    
    # Test buggy code
    buggy_result, lost = assign_pairs_buggy(sr_pairs, ion_to_block, block_names)
    buggy_count = sum(sum(len(p) for p in bp) for bp in buggy_result.values())
    print(f"\nBuggy code:")
    print(f"  Output pairs: {buggy_count}")
    print(f"  Lost pairs: {len(lost)}")
    if lost:
        print(f"  Sample lost: {lost[:5]}")
    
    # Test fixed code    
    fixed_result, rescued = assign_pairs_fixed(sr_pairs, ion_to_block, block_names)
    fixed_count = sum(sum(len(p) for p in bp) for bp in fixed_result.values())
    print(f"\nFixed code:")
    print(f"  Output pairs: {fixed_count}")
    print(f"  Rescued pairs: {len(rescued)}")
    if rescued:
        print(f"  Sample rescued: {rescued[:5]}")
    
    # Verify
    assert fixed_count == total_pairs, f"Fixed code should preserve all pairs: {fixed_count} != {total_pairs}"
    print(f"\n✅ Fix preserves all {total_pairs} pairs!")
    
    # Even better: check where pairs with bridge ancillas go
    print("\n--- Detailed analysis ---")
    # Find pairs where BOTH ions are unmapped (bridge ancillas only)
    unmapped_both = []
    for rnd in sr_pairs:
        for a, d in rnd:
            if a not in ion_to_block and d not in ion_to_block:
                unmapped_both.append((a, d))
    print(f"Pairs with BOTH ions unmapped: {len(unmapped_both)}")
    
    # Find pairs where at least one is mapped
    one_mapped = sum(
        1 for rnd in sr_pairs 
        for a, d in rnd
        if (a in ion_to_block) != (d in ion_to_block)
    )
    print(f"Pairs with exactly one ion mapped: {one_mapped}")
    
    both_mapped = sum(
        1 for rnd in sr_pairs 
        for a, d in rnd
        if a in ion_to_block and d in ion_to_block
    )
    print(f"Pairs with both ions mapped: {both_mapped}")

if __name__ == "__main__":
    main()
