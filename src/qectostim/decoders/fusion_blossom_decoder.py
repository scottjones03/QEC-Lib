from qectostim.decoders.base import Decoder
import stim 
import numpy as np
import math
from typing import Any, List
from dataclasses import dataclass, field


@dataclass
class FusionBlossomDecoder(Decoder):
    """MWPM decoder using the fusion-blossom library on Stim DEMs.
    
    Builds a matching graph from the detector error model and uses
    fusion-blossom's efficient MWPM solver. Tracks observable flips
    through matched edges.
    
    Parameters
    ----------
    dem : stim.DetectorErrorModel
        The detector error model to decode.
    """
    
    dem: stim.DetectorErrorModel
    _solver: Any = field(default=None, init=False, repr=False)
    _edge_obs_masks: List[int] = field(default_factory=list, init=False, repr=False)
    
    def __post_init__(self) -> None:
        try:
            import fusion_blossom as fb
        except ImportError as exc:
            raise ImportError(
                "FusionBlossomDecoder requires `fusion-blossom` package. "
                "Install it via `pip install fusion-blossom`."
            ) from exc
        
        self._fb = fb
        self.num_detectors = self.dem.num_detectors
        self.num_observables = self.dem.num_observables
        
        # Parse DEM to extract edges with weights and observable masks
        edges = []
        for instruction in self.dem.flattened():
            if instruction.type == 'error':
                prob = instruction.args_copy()[0]
                weight = -math.log(prob / (1 - prob)) if 0 < prob < 0.5 else 0.0
                
                dets = []
                obs_mask = 0
                for target in instruction.targets_copy():
                    if target.is_relative_detector_id():
                        dets.append(target.val)
                    elif target.is_logical_observable_id():
                        obs_mask ^= (1 << target.val)
                
                if len(dets) == 2:
                    edges.append((dets[0], dets[1], weight, obs_mask))
                elif len(dets) == 1:
                    # Boundary edge
                    edges.append((dets[0], -1, weight, obs_mask))
        
        # Build weighted edge list for fusion_blossom
        boundary_vertex = self.num_detectors
        weighted_edges = []
        self._edge_obs_masks = []
        
        for det1, det2, weight, obs_mask in edges:
            # fusion_blossom requires even integer weights
            scaled_weight = max(2, int(weight * 1000) * 2)
            if det2 == -1:
                weighted_edges.append((det1, boundary_vertex, scaled_weight))
            else:
                weighted_edges.append((det1, det2, scaled_weight))
            self._edge_obs_masks.append(obs_mask)
        
        # Create solver
        initializer = fb.SolverInitializer(
            vertex_num=self.num_detectors + 1,
            weighted_edges=weighted_edges,
            virtual_vertices=[boundary_vertex],
        )
        self._solver = fb.SolverSerial(initializer)

    def decode_batch(self, dets: np.ndarray) -> np.ndarray:
        dets = np.asarray(dets, dtype=np.uint8)
        if dets.ndim == 1:
            dets = dets.reshape(1, -1)
        
        if dets.shape[1] != self.num_detectors:
            raise ValueError(
                f"FusionBlossomDecoder expected input shape (shots, {self.num_detectors})."
            )
        
        shots = dets.shape[0]
        corrections = np.zeros((shots, self.num_observables), dtype=np.uint8)
        
        for i in range(shots):
            # Get triggered detector indices
            triggered = [j for j in range(self.num_detectors) if dets[i, j]]
            
            self._solver.clear()
            syndrome = self._fb.SyndromePattern(triggered)
            self._solver.solve(syndrome)
            
            # Get matched edges and compute observable flips
            subgraph = self._solver.subgraph()
            obs_flip = 0
            for edge_idx in subgraph:
                if edge_idx < len(self._edge_obs_masks):
                    obs_flip ^= self._edge_obs_masks[edge_idx]
            
            # Convert to array
            for obs_idx in range(self.num_observables):
                if obs_flip & (1 << obs_idx):
                    corrections[i, obs_idx] = 1
        
        return corrections