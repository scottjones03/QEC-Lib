import stim
import numpy as np
import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Union, Any, Type, Callable
from qectostim.noise.models import NoiseModel, CircuitDepolarizingNoise
from qectostim.experiments.experiment import Experiment
from qectostim.codes.abstract_css import CSSCode
from qectostim.codes.composite.concatenated import ConcatenatedCSSCode as LibraryConcatenatedCSSCode
from qectostim.codes.abstract_code import PauliString


def _array_to_pauli_string(arr: np.ndarray, pauli_type: str='Z') ->PauliString:
    return {i: pauli_type for i, v in enumerate(arr) if v}


def _pauli_string_to_array(pauli: PauliString, n: int, pauli_type: str='Z'
    ) ->np.ndarray:
    arr = np.zeros(n, dtype=np.int64)
    if isinstance(pauli, str):
        for i, op in enumerate(pauli):
            if i < n and (op == pauli_type or op == 'Y'):
                arr[i] = 1
        return arr
    for i, op in pauli.items():
        if op == pauli_type or op == 'Y':
            arr[i] = 1
    return arr


def _get_code_distance(code) ->int:
    if hasattr(code, 'd'):
        return code.d
    if hasattr(code, 'distance'):
        return code.distance
    if hasattr(code, 'metadata'):
        meta = code.metadata
        if isinstance(meta, dict):
            return meta.get('distance', meta.get('d', 3))
    return 3


def _get_code_hz(code) ->np.ndarray:
    hz = getattr(code, 'Hz', None)
    if hz is not None:
        return hz
    return getattr(code, 'hz', None)


def _get_code_hx(code) ->np.ndarray:
    hx = getattr(code, 'Hx', None)
    if hx is not None:
        return hx
    return getattr(code, 'hx', None)


def _get_code_k(code) ->int:
    if hasattr(code, 'k'):
        return code.k
    if hasattr(code, 'metadata'):
        meta = code.metadata
        if isinstance(meta, dict):
            return meta.get('k', 1)
    return 1


def _get_code_transversal_block_count(code) ->Optional[int]:
    if hasattr(code, 'transversal_block_count'):
        return code.transversal_block_count
    if hasattr(code, 'metadata'):
        meta = code.metadata
        if isinstance(meta, dict):
            return meta.get('transversal_block_count')
    return None


def _get_code_lz(code) ->np.ndarray:
    if hasattr(code, 'Lz'):
        lz = code.Lz
        return lz if lz.ndim == 1 else lz[0]
    elif hasattr(code, 'lz'):
        lz = code.lz
        return lz if lz.ndim == 1 else lz[0]
    elif hasattr(code, 'logical_z'):
        lz_list = code.logical_z
        if lz_list:
            return _pauli_string_to_array(lz_list[0], code.n, 'Z')
    elif hasattr(code, '_logical_z'):
        lz_list = code._logical_z
        if lz_list:
            return _pauli_string_to_array(lz_list[0], code.n, 'Z')
    return np.zeros(code.n, dtype=np.int64)


def _get_code_lz_info(code) ->Tuple[List[int], str]:
    if hasattr(code, 'metadata') and isinstance(code.metadata, dict):
        meta = code.metadata
        if 'lz_pauli_type' in meta and 'lz_support' in meta:
            return meta['lz_support'], meta['lz_pauli_type']
    lz_obj = None
    if hasattr(code, 'logical_z') and code.logical_z:
        lz_obj = code.logical_z[0]
    elif hasattr(code, '_logical_z') and code._logical_z:
        lz_obj = code._logical_z[0]
    if lz_obj is None:
        lz_arr = _get_code_lz(code)
        return [i for i in range(len(lz_arr)) if lz_arr[i] == 1], 'Z'
    if isinstance(lz_obj, dict):
        z_support = [i for i, p in lz_obj.items() if p == 'Z']
        x_support = [i for i, p in lz_obj.items() if p == 'X']
        if z_support:
            return sorted(z_support), 'Z'
        elif x_support:
            return sorted(x_support), 'X'
        else:
            lz_arr = _get_code_lz(code)
            return [i for i in range(len(lz_arr)) if lz_arr[i] == 1], 'Z'
    if isinstance(lz_obj, str):
        z_support = [i for i, c in enumerate(lz_obj) if c == 'Z']
        x_support = [i for i, c in enumerate(lz_obj) if c == 'X']
        if z_support:
            return z_support, 'Z'
        elif x_support:
            return x_support, 'X'
        else:
            lz_arr = _get_code_lz(code)
            return [i for i in range(len(lz_arr)) if lz_arr[i] == 1], 'Z'
    lz_arr = _get_code_lz(code)
    return [i for i in range(len(lz_arr)) if lz_arr[i] == 1], 'Z'


def _get_code_lx_info(code) ->Tuple[List[int], str]:
    if hasattr(code, 'metadata') and isinstance(code.metadata, dict):
        meta = code.metadata
        if 'lx_pauli_type' in meta and 'lx_support' in meta:
            return meta['lx_support'], meta['lx_pauli_type']
    lx_obj = None
    if hasattr(code, 'logical_x') and code.logical_x:
        lx_obj = code.logical_x[0]
    elif hasattr(code, '_logical_x') and code._logical_x:
        lx_obj = code._logical_x[0]
    if lx_obj is None:
        lx_arr = _get_code_lx(code)
        return [i for i in range(len(lx_arr)) if lx_arr[i] == 1], 'X'
    if isinstance(lx_obj, dict):
        x_support = [i for i, p in lx_obj.items() if p == 'X']
        z_support = [i for i, p in lx_obj.items() if p == 'Z']
        if x_support:
            return sorted(x_support), 'X'
        elif z_support:
            return sorted(z_support), 'Z'
        else:
            return [], 'X'
    if isinstance(lx_obj, str):
        x_support = [i for i, c in enumerate(lx_obj) if c == 'X']
        z_support = [i for i, c in enumerate(lx_obj) if c == 'Z']
        if x_support:
            return x_support, 'X'
        elif z_support:
            return z_support, 'Z'
        else:
            return [], 'X'
    lx_arr = _get_code_lx(code)
    return [i for i in range(len(lx_arr)) if lx_arr[i] == 1], 'X'


def _get_code_lx(code) ->np.ndarray:
    if hasattr(code, 'Lx'):
        lx = code.Lx
        return lx if lx.ndim == 1 else lx[0]
    elif hasattr(code, 'lx'):
        lx = code.lx
        return lx if lx.ndim == 1 else lx[0]
    elif hasattr(code, 'logical_x'):
        lx_list = code.logical_x
        if lx_list:
            return _pauli_string_to_array(lx_list[0], code.n, 'X')
    elif hasattr(code, '_logical_x'):
        lx_list = code._logical_x
        if lx_list:
            return _pauli_string_to_array(lx_list[0], code.n, 'X')
    return np.zeros(code.n, dtype=np.int64)


def _get_code_swap_after_h(code) ->list:
    if hasattr(code, 'swap_after_h'):
        return code.swap_after_h or []
    if hasattr(code, 'metadata') and isinstance(code.metadata, dict):
        return code.metadata.get('swap_after_h', [])
    return []


def _get_code_swap_after_h_l2(code) ->list:
    if hasattr(code, 'swap_after_h_l2'):
        return code.swap_after_h_l2 or []
    if hasattr(code, 'metadata') and isinstance(code.metadata, dict):
        return code.metadata.get('swap_after_h_l2', [])
    return []


def _get_inner_logical_z(concat_code) ->np.ndarray:
    inner_code = concat_code.code_at_level(0) if hasattr(concat_code,
        'code_at_level') else concat_code.base_code
    return _get_code_lz(inner_code)


def _get_inner_logical_x(concat_code) ->np.ndarray:
    inner_code = concat_code.code_at_level(0) if hasattr(concat_code,
        'code_at_level') else concat_code.base_code
    return _get_code_lx(inner_code)


def get_effective_decoder_type(code) ->str:
    dt = code._metadata.get('decoder_type', 'syndrome')
    if dt == 'auto':
        k = _get_code_k(code)
        return 'parity' if k >= 2 else 'syndrome'
    return dt


def make_css_code(name: str, n: int, k: int, d: int, Hz: np.ndarray, Hx: np
    .ndarray, logical_z_ops: List[np.ndarray], logical_x_ops: List[np.
    ndarray], h_qubits: Optional[List[int]]=None, logical_h_qubits:
    Optional[List[int]]=None, plus_h_qubits: Optional[List[int]]=None,
    plus_encoding_cnots: Optional[List[Tuple[int, int]]]=None,
    plus_encoding_cnot_rounds: Optional[List[List[Tuple[int, int]]]]=None,
    pre_h_cnots: Optional[List[Tuple[int, int]]]=None, encoding_cnots:
    Optional[List[Tuple[int, int]]]=None, encoding_cnot_rounds: Optional[
    List[List[Tuple[int, int]]]]=None, verification_qubits: Optional[List[
    int]]=None, swap_after_h: Optional[List[Tuple[int, int]]]=None,
    swap_after_h_l2: Optional[List[Tuple[int, int]]]=None,
    uses_bellpair_prep: bool=False, idle_schedule: Optional[Dict[str, List[
    int]]]=None, transversal_block_count: Optional[int]=None, decoder_type:
    str='auto', outer_decoder_type: Optional[str]=None, post_selection_type:
    str='auto', uses_edt: bool=False, skip_validation: bool=False) ->CSSCode:
    logical_z = [_array_to_pauli_string(lz, 'Z') for lz in logical_z_ops]
    logical_x = [_array_to_pauli_string(lx, 'X') for lx in logical_x_ops]
    if encoding_cnot_rounds is None and encoding_cnots:
        encoding_cnot_rounds = [[(c, t)] for c, t in encoding_cnots]
    effective_decoder_type = decoder_type
    if decoder_type == 'auto':
        effective_decoder_type = 'parity' if k >= 2 else 'syndrome'
    effective_post_selection_type = post_selection_type
    if post_selection_type == 'auto':
        if k >= 2 and d <= 2:
            effective_post_selection_type = 'parity'
        elif verification_qubits:
            effective_post_selection_type = 'verification'
        else:
            effective_post_selection_type = 'parity'
    effective_uses_edt = uses_edt
    if k >= 2 and d <= 2 and not uses_edt:
        effective_uses_edt = True
    effective_logical_h_qubits = logical_h_qubits
    if logical_h_qubits is None:
        effective_logical_h_qubits = list(h_qubits) if h_qubits else list(range
            (n))
    metadata: Dict[str, Any] = {'name': name, 'k': k, 'd': d, 'h_qubits': 
        h_qubits or [], 'logical_h_qubits': effective_logical_h_qubits,
        'plus_h_qubits': plus_h_qubits, 'plus_encoding_cnots':
        plus_encoding_cnots, 'plus_encoding_cnot_rounds':
        plus_encoding_cnot_rounds, 'pre_h_cnots': pre_h_cnots or [],
        'encoding_cnots': encoding_cnots or [], 'encoding_cnot_rounds':
        encoding_cnot_rounds, 'verification_qubits': verification_qubits or
        [], 'swap_after_h': swap_after_h or [], 'swap_after_h_l2': 
        swap_after_h_l2 or [], 'uses_bellpair_prep': uses_bellpair_prep,
        'idle_schedule': idle_schedule, 'transversal_block_count':
        transversal_block_count, 'decoder_type': effective_decoder_type,
        'outer_decoder_type': outer_decoder_type, 'post_selection_type':
        effective_post_selection_type, 'uses_edt': effective_uses_edt}
    return CSSCode(hx=Hx, hz=Hz, logical_x=logical_x, logical_z=logical_z,
        metadata=metadata, skip_validation=skip_validation)


@dataclass
class ConcatenatedCode:
    levels: List[CSSCode]
    name: Optional[str] = None
    custom_decoder_fn: Optional[Callable] = None
    custom_accept_l2_fn: Optional[Callable] = None
    custom_post_selection_l2_fn: Optional[Callable] = None

    def __post_init__(self):
        if self.name is None:
            self.name = 'Concat[' + '->'.join(c.name for c in self.levels
                ) + ']'

    @property
    def num_levels(self) ->int:
        return len(self.levels)

    @property
    def total_qubits(self) ->int:
        result = 1
        for code in self.levels:
            result *= code.n
        return result

    @property
    def inner_code(self) ->CSSCode:
        return self.levels[0]

    @property
    def outer_code(self) ->CSSCode:
        return self.levels[-1]

    @property
    def n(self) ->int:
        return self.total_qubits

    @property
    def k(self) ->int:
        return self.levels[-1].k if self.levels else 1

    def qubits_at_level(self, level: int) ->int:
        result = 1
        for i in range(level + 1):
            result *= self.levels[i].n
        return result

    def code_at_level(self, level: int) ->CSSCode:
        return self.levels[level]

    @property
    def has_custom_l2_acceptance(self) ->bool:
        return self.custom_accept_l2_fn is not None

    @property
    def has_custom_decoder(self) ->bool:
        return self.custom_decoder_fn is not None

    @property
    def has_custom_post_selection(self) ->bool:
        return self.custom_post_selection_l2_fn is not None


@dataclass
class FTVerificationResult:
    verification_method: str
    detector_ranges: List[List[int]]
    accepted_loc: int
    num_copies_used: int
    num_verification_rounds: int
    all_trivial_condition: str

    def to_dict(self) ->Dict:
        return {'verification_method': self.verification_method,
            'detector_info': self.detector_ranges, 'accepted_loc': self.
            accepted_loc, 'num_copies_used': self.num_copies_used,
            'num_verification_rounds': self.num_verification_rounds,
            'verification_outcomes': self.detector_ranges}

    @classmethod
    def from_shor_result(cls, syndromes: List[Dict], data_loc: int,
        num_rounds: int, detector_counter_start: int) ->'FTVerificationResult':
        detector_ranges = []
        for round_info in syndromes:
            for stab_type in ['X_syndromes', 'Z_syndromes']:
                if stab_type in round_info:
                    for stab_detectors in round_info[stab_type]:
                        if stab_detectors:
                            start = min(stab_detectors)
                            end = max(stab_detectors) + 1
                            detector_ranges.append([start, end])
        return cls(verification_method='shor', detector_ranges=
            detector_ranges, accepted_loc=data_loc, num_copies_used=1,
            num_verification_rounds=num_rounds, all_trivial_condition=
            'all syndrome measurements zero')

    @classmethod
    def from_steane_result(cls, comparisons: List, kept_loc: int,
        num_copies: int) ->'FTVerificationResult':
        detector_ranges = []

        def extract_ranges(obj):
            if isinstance(obj, dict):
                for val in obj.values():
                    extract_ranges(val)
            elif isinstance(obj, list):
                if len(obj) == 2 and isinstance(obj[0], int) and isinstance(obj
                    [1], int):
                    detector_ranges.append(obj)
                elif obj and isinstance(obj[0], int):
                    start = min(obj)
                    end = max(obj) + 1
                    detector_ranges.append([start, end])
                else:
                    for item in obj:
                        extract_ranges(item)
        extract_ranges(comparisons)
        return cls(verification_method='steane', detector_ranges=
            detector_ranges, accepted_loc=kept_loc, num_copies_used=
            num_copies, num_verification_rounds=2, all_trivial_condition=
            'all copy comparisons consistent')


@dataclass
class PauliFrame:
    x_corrections: np.ndarray
    z_corrections: np.ndarray
    outer_x: int = 0
    outer_z: int = 0

    @classmethod
    def for_l1(cls, k: int=1) ->'PauliFrame':
        return cls(x_corrections=np.zeros(1, dtype=int), z_corrections=np.
            zeros(1, dtype=int))

    @classmethod
    def for_l2(cls, n: int, k: int=1) ->'PauliFrame':
        return cls(x_corrections=np.zeros(n, dtype=int), z_corrections=np.
            zeros(n, dtype=int))

    @classmethod
    def from_prep_outcomes(cls, sample: np.ndarray, prep_meas_indices: Dict,
        logical_z: np.ndarray, logical_x: np.ndarray, n: int) ->'PauliFrame':
        frame = cls(x_corrections=np.zeros(n, dtype=int), z_corrections=np.
            zeros(n, dtype=int))
        if prep_meas_indices and 'Lz_meas_indices' in prep_meas_indices:
            lz_indices = prep_meas_indices['Lz_meas_indices']
            if isinstance(lz_indices, list) and len(lz_indices) > 0:
                if isinstance(lz_indices[0], list):
                    inner_lz_values = np.zeros(n, dtype=int)
                    for i, idx_range in enumerate(lz_indices):
                        if i >= n:
                            break
                        if isinstance(idx_range, list) and len(idx_range) >= 2:
                            m_data = np.array(sample[idx_range[0]:idx_range
                                [1]], dtype=int)
                            inner_lz_values[i] = int(np.dot(logical_z,
                                m_data) % 2)
                    outer_lz = int(np.dot(logical_z, inner_lz_values) % 2)
                    if outer_lz == 1:
                        frame.outer_x = 1
        return frame

    def apply_x_correction(self, block_idx: int, value: int):
        if block_idx < len(self.x_corrections):
            self.x_corrections[block_idx] = (self.x_corrections[block_idx] +
                value) % 2

    def apply_z_correction(self, block_idx: int, value: int):
        if block_idx < len(self.z_corrections):
            self.z_corrections[block_idx] = (self.z_corrections[block_idx] +
                value) % 2

    def apply_outer_x(self, value: int):
        self.outer_x = (self.outer_x + value) % 2

    def apply_outer_z(self, value: int):
        self.outer_z = (self.outer_z + value) % 2

    def apply_h_gate(self, block_idx: int=None):
        if block_idx is None:
            self.x_corrections, self.z_corrections = self.z_corrections.copy(
                ), self.x_corrections.copy()
            self.outer_x, self.outer_z = self.outer_z, self.outer_x
        elif block_idx < len(self.x_corrections):
            self.x_corrections[block_idx], self.z_corrections[block_idx
                ] = self.z_corrections[block_idx], self.x_corrections[block_idx
                ]

    def apply_cnot(self, control_idx: int, target_idx: int):
        if control_idx < len(self.x_corrections) and target_idx < len(self.
            x_corrections):
            self.x_corrections[target_idx] = (self.x_corrections[target_idx
                ] + self.x_corrections[control_idx]) % 2
            self.z_corrections[control_idx] = (self.z_corrections[
                control_idx] + self.z_corrections[target_idx]) % 2

    def get_z_basis_correction(self) ->int:
        return self.outer_x

    def get_x_basis_correction(self) ->int:
        return self.outer_z


@dataclass
class KnillECResult:
    prep_detectors: List[int]
    prep_detectors_l2: List[int]
    detector_X: List[List[int]]
    detector_Z: List[List[int]]
    measurement_X: List[List[int]] = field(default_factory=list)
    measurement_Z: List[List[int]] = field(default_factory=list)
    output_location: int = 0
    gauge_syndrome_z: Optional[List[int]] = None
    gauge_syndrome_x: Optional[List[int]] = None

    @classmethod
    def from_tuple_l1(cls, result: Tuple) ->'KnillECResult':
        output_loc = result[3] if len(result) > 3 else 0
        meas_X = result[4] if len(result) > 4 else []
        meas_Z = result[5] if len(result) > 5 else []
        return cls(prep_detectors=result[0], prep_detectors_l2=[],
            detector_X=[result[2]] if not isinstance(result[2], list) or 
            result[2] and isinstance(result[2][0], int) else result[2],
            detector_Z=[result[1]] if not isinstance(result[1], list) or 
            result[1] and isinstance(result[1][0], int) else result[1],
            measurement_X=[meas_X] if meas_X and isinstance(meas_X[0], int)
             else meas_X, measurement_Z=[meas_Z] if meas_Z and isinstance(
            meas_Z[0], int) else meas_Z, output_location=output_loc)

    @classmethod
    def from_tuple_l2(cls, result: Tuple) ->'KnillECResult':
        output_loc = result[4] if len(result) > 4 else 0
        meas_X = result[5] if len(result) > 5 else []
        meas_Z = result[6] if len(result) > 6 else []
        return cls(prep_detectors=result[0], prep_detectors_l2=result[1],
            detector_X=result[3] if isinstance(result[3], list) else [
            result[3]], detector_Z=result[2] if isinstance(result[2], list)
             else [result[2]], measurement_X=[meas_X] if meas_X and
            isinstance(meas_X[0], int) else meas_X, measurement_Z=[meas_Z] if
            meas_Z and isinstance(meas_Z[0], int) else meas_Z,
            output_location=output_loc)

    def compute_gauge_syndromes(self, sample: np.ndarray, check_matrix_z:
        np.ndarray, check_matrix_x: np.ndarray) ->'KnillECResult':

        def compute_syndrome_int(m: np.ndarray, H: np.ndarray) ->int:
            syndrome = 0
            for i in range(H.shape[0]):
                parity = int(np.sum(m * H[i, :]) % 2)
                syndrome += parity * (1 << i)
            return syndrome
        if self.measurement_Z:
            outer_meas_z = self.measurement_Z[-1
                ] if self.measurement_Z else None
            if outer_meas_z and isinstance(outer_meas_z, list):
                if len(outer_meas_z) > 0 and isinstance(outer_meas_z[0], list):
                    self.gauge_syndrome_z = []
                    for meas_range in outer_meas_z:
                        if isinstance(meas_range, list) and len(meas_range
                            ) >= 2:
                            m_data = np.array(sample[meas_range[0]:
                                meas_range[1]], dtype=int)
                            gauge = compute_syndrome_int(m_data, check_matrix_z
                                )
                            self.gauge_syndrome_z.append(gauge)
                elif len(outer_meas_z) >= 2 and isinstance(outer_meas_z[0], int
                    ):
                    m_data = np.array(sample[outer_meas_z[0]:outer_meas_z[1
                        ]], dtype=int)
                    self.gauge_syndrome_z = [compute_syndrome_int(m_data,
                        check_matrix_z)]
        if self.measurement_X:
            outer_meas_x = self.measurement_X[-1
                ] if self.measurement_X else None
            if outer_meas_x and isinstance(outer_meas_x, list):
                if len(outer_meas_x) > 0 and isinstance(outer_meas_x[0], list):
                    self.gauge_syndrome_x = []
                    for meas_range in outer_meas_x:
                        if isinstance(meas_range, list) and len(meas_range
                            ) >= 2:
                            m_data = np.array(sample[meas_range[0]:
                                meas_range[1]], dtype=int)
                            gauge = compute_syndrome_int(m_data, check_matrix_x
                                )
                            self.gauge_syndrome_x.append(gauge)
                elif len(outer_meas_x) >= 2 and isinstance(outer_meas_x[0], int
                    ):
                    m_data = np.array(sample[outer_meas_x[0]:outer_meas_x[1
                        ]], dtype=int)
                    self.gauge_syndrome_x = [compute_syndrome_int(m_data,
                        check_matrix_x)]
        return self


@dataclass
class PauliFrameUpdate:
    x_correction_source: Optional[str] = None
    z_correction_source: Optional[str] = None
    source_detectors: List = field(default_factory=list)
    target_block: Optional[int] = None

    def requires_x_correction(self) ->bool:
        return self.x_correction_source is not None

    def requires_z_correction(self) ->bool:
        return self.z_correction_source is not None

    def to_dict(self) ->Dict:
        return {'x_correction_source': self.x_correction_source,
            'z_correction_source': self.z_correction_source,
            'source_detectors': self.source_detectors, 'target_block': self
            .target_block}


@dataclass
class GateResult:
    gate_type: str
    implementation: str
    level: int
    detectors: List = field(default_factory=list)
    pauli_frame_update: Optional[PauliFrameUpdate] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class PrepResult:
    level: int
    detector_0prep: List = field(default_factory=list)
    detector_0prep_l2: Optional[List] = None
    detector_X: List = field(default_factory=list)
    detector_Z: List = field(default_factory=list)
    children: List = field(default_factory=list)


@dataclass
class ECResult:
    level: int
    ec_type: str
    detector_0prep: List = field(default_factory=list)
    detector_0prep_l2: Optional[List] = None
    detector_X: List = field(default_factory=list)
    detector_Z: List = field(default_factory=list)
    children: List = field(default_factory=list)


def derive_css_encoding_circuit(Hz: np.ndarray, Hx: np.ndarray, logical_x:
    np.ndarray, logical_z: np.ndarray) ->Tuple[List[int], List[Tuple[int,
    int]], List[List[Tuple[int, int]]]]:
    n = Hz.shape[1]
    m = Hz.shape[0]
    h_qubits = [i for i in range(n) if logical_x[i] == 1]
    encoding_cnots = []
    h_set = set(h_qubits)
    non_h = [i for i in range(n) if i not in h_set]
    for row in Hz:
        support = [i for i in range(n) if row[i] == 1]
        sources = [i for i in support if i in h_set]
        targets = [i for i in support if i not in h_set]
        if sources and targets:
            for tgt in targets:
                src = min(sources, key=lambda s: abs(s - tgt))
                cnot = src, tgt
                if cnot not in encoding_cnots:
                    encoding_cnots.append(cnot)
    if len(encoding_cnots) < n - len(h_qubits):
        encoding_cnots = []
        if h_qubits:
            src = h_qubits[0]
            for tgt in non_h:
                encoding_cnots.append((src, tgt))
    encoding_cnot_rounds = [[cnot] for cnot in encoding_cnots]
    return h_qubits, encoding_cnots, encoding_cnot_rounds


def derive_steane_style_encoding(n: int, Hz: np.ndarray, logical_x: np.ndarray
    ) ->Tuple[List[int], List[Tuple[int, int]]]:
    h_qubits = [i for i in range(n) if logical_x[i] == 1]
    h_set = set(h_qubits)
    encoding_cnots = []
    covered_targets = set()
    for row in Hz:
        support = [i for i in range(n) if row[i] == 1]
        sources = [i for i in support if i in h_set]
        targets = [i for i in support if i not in h_set and i not in
            covered_targets]
        if sources and targets:
            for tgt in targets:
                src = sources[0]
                encoding_cnots.append((src, tgt))
                covered_targets.add(tgt)
    remaining = [i for i in range(n) if i not in h_set and i not in
        covered_targets]
    if remaining and h_qubits:
        src = h_qubits[0]
        for tgt in remaining:
            encoding_cnots.append((src, tgt))
    return h_qubits, encoding_cnots


def create_concatenated_code(codes: List[CSSCode]) ->ConcatenatedCode:
    return ConcatenatedCode(levels=codes)


class PhysicalOps:

    @staticmethod
    def reset(circuit: stim.Circuit, loc: int, n: int):
        for i in range(n):
            circuit.append('R', loc + i)

    @staticmethod
    def noisy_reset(circuit: stim.Circuit, loc: int, n: int, p: float):
        for i in range(n):
            circuit.append('R', loc + i)
        for i in range(n):
            circuit.append('X_ERROR', loc + i, p)

    @staticmethod
    def h(circuit: stim.Circuit, loc: int):
        circuit.append('H', loc)

    @staticmethod
    def cnot(circuit: stim.Circuit, ctrl: int, targ: int):
        circuit.append('CNOT', [ctrl, targ])

    @staticmethod
    def noisy_cnot(circuit: stim.Circuit, ctrl: int, targ: int, p: float):
        circuit.append('CNOT', [ctrl, targ])
        circuit.append('DEPOLARIZE2', [ctrl, targ], p)

    @staticmethod
    def swap(circuit: stim.Circuit, q1: int, q2: int):
        circuit.append('SWAP', [q1, q2])

    @staticmethod
    def measure(circuit: stim.Circuit, loc: int):
        circuit.append('M', loc)

    @staticmethod
    def noisy_measure(circuit: stim.Circuit, loc: int, p: float):
        circuit.append('X_ERROR', loc, p)
        circuit.append('M', loc)

    @staticmethod
    def detector(circuit: stim.Circuit, offset: int):
        circuit.append('DETECTOR', stim.target_rec(offset))

    @staticmethod
    def depolarize1(circuit: stim.Circuit, loc: int, p: float):
        circuit.append('DEPOLARIZE1', loc, p)


class TransversalOps:

    def __init__(self, concat_code: ConcatenatedCode):
        self.concat_code = concat_code

    def block_size(self, level: int) ->int:
        return self.concat_code.qubits_at_level(level)

    def _get_inner_code(self) ->Optional[CSSCode]:
        if self.concat_code.num_levels > 0:
            return self.concat_code.code_at_level(0)
        return None

    def append_h(self, circuit: stim.Circuit, loc: int, N_prev: int, N_now:
        int, level: int=0):
        if N_prev == 1:
            for i in range(N_now):
                PhysicalOps.h(circuit, loc + i)
            code = self._get_inner_code()
            swap_pattern = _get_code_swap_after_h(code
                ) if code is not None else []
            if swap_pattern and N_now == code.n:
                for q1, q2 in swap_pattern:
                    PhysicalOps.swap(circuit, loc + q1, loc + q2)
        else:
            for i in range(N_now):
                self.append_h(circuit, (loc + i) * N_prev, 1, N_prev, level)
            if self.concat_code.num_levels > 1:
                outer_code = self.concat_code.code_at_level(1
                    ) if level == 0 else None
                if outer_code is not None and N_now == outer_code.n:
                    swap_l2 = _get_code_swap_after_h_l2(outer_code)
                    swap_l1 = _get_code_swap_after_h(outer_code)
                    swap_pattern = swap_l2 if swap_l2 else swap_l1
                    if swap_pattern:
                        for q1, q2 in swap_pattern:
                            for j in range(N_prev):
                                PhysicalOps.swap(circuit, (loc + q1) *
                                    N_prev + j, (loc + q2) * N_prev + j)

    def append_logical_h(self, circuit: stim.Circuit, loc: int, N_prev: int,
        N_now: int, code: CSSCode=None, level: int=0):
        if code is None:
            code = self._get_inner_code()
        if code is None or code.logical_h_qubits is None or len(code.
            logical_h_qubits) == N_now:
            self.append_h(circuit, loc, N_prev, N_now, level)
        elif N_prev == 1:
            for q in code.logical_h_qubits:
                PhysicalOps.h(circuit, loc + q)
            if code.swap_after_h and N_now == code.n:
                for q1, q2 in code.swap_after_h:
                    PhysicalOps.swap(circuit, loc + q1, loc + q2)
        else:
            for i in range(N_now):
                self.append_logical_h(circuit, (loc + i) * N_prev, 1,
                    N_prev, code, level)

    def append_cnot(self, circuit: stim.Circuit, loc1: int, loc2: int,
        N_prev: int, N_now: int):
        N = N_prev * N_now
        for i in range(N):
            PhysicalOps.cnot(circuit, loc1 * N_prev + i, loc2 * N_prev + i)

    def append_noisy_cnot(self, circuit: stim.Circuit, loc1: int, loc2: int,
        N_prev: int, N_now: int, p: float):
        N = N_prev * N_now
        for i in range(N):
            PhysicalOps.cnot(circuit, loc1 * N_prev + i, loc2 * N_prev + i)
        if p > 0:
            for i in range(N):
                circuit.append('DEPOLARIZE2', [loc1 * N_prev + i, loc2 *
                    N_prev + i], p)

    def append_cz(self, circuit: stim.Circuit, loc1: int, loc2: int, N_prev:
        int, N_now: int):
        N = N_prev * N_now
        for i in range(N):
            circuit.append('CZ', [loc1 * N_prev + i, loc2 * N_prev + i])

    def append_noisy_cz(self, circuit: stim.Circuit, loc1: int, loc2: int,
        N_prev: int, N_now: int, p: float):
        N = N_prev * N_now
        for i in range(N):
            circuit.append('CZ', [loc1 * N_prev + i, loc2 * N_prev + i])
        if p > 0:
            for i in range(N):
                circuit.append('DEPOLARIZE2', [loc1 * N_prev + i, loc2 *
                    N_prev + i], p)

    def append_m_x(self, circuit: stim.Circuit, loc: int, N_prev: int,
        N_now: int, detector_counter: List[int], code: CSSCode=None) ->List:
        if N_prev == 1:
            for i in range(N_now):
                circuit.append('MX', [loc + i])
            for i in range(N_now):
                PhysicalOps.detector(circuit, i - N_now)
            detector_m = [detector_counter[0], detector_counter[0] + N_now]
            detector_counter[0] += N_now
        else:
            inner_code = self._get_inner_code() if N_prev > 1 else None
            detector_m = [self.append_m_x(circuit, (loc + i) * N_prev, 1,
                N_prev, detector_counter, inner_code) for i in range(N_now)]
        return detector_m

    def append_noisy_m_x(self, circuit: stim.Circuit, loc: int, N_prev: int,
        N_now: int, p: float, detector_counter: List[int], code: CSSCode=
        None, measurement_counter: List[int]=None) ->Tuple:
        if N_prev == 1:
            meas_start = measurement_counter[0
                ] if measurement_counter else None
            for i in range(N_now):
                PhysicalOps.depolarize1(circuit, loc + i, p)
                circuit.append('MX', [loc + i])
            if measurement_counter is not None:
                measurement_counter[0] += N_now
            for i in range(N_now):
                PhysicalOps.detector(circuit, i - N_now)
            detector_m = [detector_counter[0], detector_counter[0] + N_now]
            detector_counter[0] += N_now
            if measurement_counter is not None:
                meas_range = [meas_start, meas_start + N_now]
                return detector_m, meas_range
            else:
                return detector_m
        else:
            inner_code = self._get_inner_code() if N_prev > 1 else None
            results = [self.append_noisy_m_x(circuit, (loc + i) * N_prev, 1,
                N_prev, p, detector_counter, inner_code,
                measurement_counter) for i in range(N_now)]
            if measurement_counter is not None:
                detector_m = [r[0] for r in results]
                meas_m = [r[1] for r in results]
                return detector_m, meas_m
            else:
                return results

    def append_raw_m_x(self, circuit: stim.Circuit, loc: int, N_prev: int,
        N_now: int, p: float, measurement_counter: List[int]) ->List[int]:
        if N_prev == 1:
            meas_start = measurement_counter[0]
            for i in range(N_now):
                PhysicalOps.depolarize1(circuit, loc + i, p)
                circuit.append('MX', [loc + i])
            measurement_counter[0] += N_now
            return [meas_start, meas_start + N_now]
        else:
            results = [self.append_raw_m_x(circuit, (loc + i) * N_prev, 1,
                N_prev, p, measurement_counter) for i in range(N_now)]
            return results

    def append_raw_m_z(self, circuit: stim.Circuit, loc: int, N_prev: int,
        N_now: int, p: float, measurement_counter: List[int]) ->List[int]:
        if N_prev == 1:
            meas_start = measurement_counter[0]
            for i in range(N_now):
                PhysicalOps.noisy_measure(circuit, loc + i, p)
            measurement_counter[0] += N_now
            return [meas_start, meas_start + N_now]
        else:
            results = [self.append_raw_m_z(circuit, (loc + i) * N_prev, 1,
                N_prev, p, measurement_counter) for i in range(N_now)]
            return results

    def append_swap(self, circuit: stim.Circuit, loc1: int, loc2: int,
        N_prev: int, N_now: int):
        for i in range(N_prev * N_now):
            PhysicalOps.swap(circuit, N_prev * loc1 + i, N_prev * loc2 + i)

    def append_m(self, circuit: stim.Circuit, loc: int, N_prev: int, N_now:
        int, detector_counter: List[int], code: CSSCode=None) ->List:
        if N_prev == 1:
            for i in range(N_now):
                PhysicalOps.measure(circuit, loc + i)
            for i in range(N_now):
                PhysicalOps.detector(circuit, i - N_now)
            detector_m = [detector_counter[0], detector_counter[0] + N_now]
            detector_counter[0] += N_now
        else:
            inner_code = self._get_inner_code() if N_prev > 1 else None
            detector_m = [self.append_m(circuit, (loc + i) * N_prev, 1,
                N_prev, detector_counter, inner_code) for i in range(N_now)]
        return detector_m

    def append_noisy_m(self, circuit: stim.Circuit, loc: int, N_prev: int,
        N_now: int, p: float, detector_counter: List[int], code: CSSCode=
        None, measurement_counter: List[int]=None) ->Tuple:
        if N_prev == 1:
            meas_start = measurement_counter[0
                ] if measurement_counter else None
            for i in range(N_now):
                PhysicalOps.noisy_measure(circuit, loc + i, p)
            if measurement_counter is not None:
                measurement_counter[0] += N_now
            for i in range(N_now):
                PhysicalOps.detector(circuit, i - N_now)
            detector_m = [detector_counter[0], detector_counter[0] + N_now]
            detector_counter[0] += N_now
            if measurement_counter is not None:
                meas_range = [meas_start, meas_start + N_now]
                return detector_m, meas_range
            else:
                return detector_m
        else:
            inner_code = self._get_inner_code() if N_prev > 1 else None
            results = [self.append_noisy_m(circuit, (loc + i) * N_prev, 1,
                N_prev, p, detector_counter, inner_code,
                measurement_counter) for i in range(N_now)]
            if measurement_counter is not None:
                detector_m = [r[0] for r in results]
                meas_m = [r[1] for r in results]
                return detector_m, meas_m
            else:
                return results

    def append_m_z(self, circuit: stim.Circuit, loc: int, N_prev: int,
        N_now: int, detector_counter: List[int], code: CSSCode=None) ->List:
        return self.append_m(circuit, loc, N_prev, N_now, detector_counter,
            code)

    def append_noisy_m_z(self, circuit: stim.Circuit, loc: int, N_prev: int,
        N_now: int, p: float, detector_counter: List[int], code: CSSCode=
        None, measurement_counter: List[int]=None) ->Tuple:
        return self.append_noisy_m(circuit, loc, N_prev, N_now, p,
            detector_counter, code, measurement_counter)

    def append_noisy_wait(self, circuit: stim.Circuit, list_loc: List[int],
        N: int, p: float, gamma: float, steps: int=1):
        ew = 3 / 4 * (1 - (1 - 4 / 3 * gamma) ** steps)
        for loc in list_loc:
            for j in range(N):
                PhysicalOps.depolarize1(circuit, loc + j, ew)


class LogicalGate(ABC):

    def __init__(self, concat_code: ConcatenatedCode):
        self.concat_code = concat_code

    @property
    @abstractmethod
    def gate_name(self) ->str:
        pass

    @property
    @abstractmethod
    def implementation_name(self) ->str:
        pass

    def block_size(self, level: int) ->int:
        return self.concat_code.qubits_at_level(level)


class LogicalHGate(LogicalGate):

    @property
    def gate_name(self) ->str:
        return 'H'

    @abstractmethod
    def apply(self, circuit: stim.Circuit, loc: int, level: int,
        detector_counter: List[int]) ->GateResult:
        pass


class LogicalCNOTGate(LogicalGate):

    @property
    def gate_name(self) ->str:
        return 'CNOT'

    @abstractmethod
    def apply(self, circuit: stim.Circuit, loc_ctrl: int, loc_targ: int,
        level: int, detector_counter: List[int]) ->GateResult:
        pass


class LogicalMeasurement(LogicalGate):

    @property
    def gate_name(self) ->str:
        return 'MEASURE'

    @abstractmethod
    def apply(self, circuit: stim.Circuit, loc: int, level: int,
        detector_counter: List[int], basis: str='z') ->GateResult:
        pass


class TransversalHGate(LogicalHGate):

    def __init__(self, concat_code: ConcatenatedCode, ops: TransversalOps):
        super().__init__(concat_code)
        self.ops = ops

    @property
    def implementation_name(self) ->str:
        return 'transversal'

    def apply(self, circuit, loc, level, detector_counter):
        N_prev = self.block_size(level - 1) if level > 0 else 1
        N_now = self.concat_code.code_at_level(level).n
        self.ops.append_h(circuit, loc, N_prev, N_now)
        return GateResult(self.gate_name, self.implementation_name, level)


class TeleportationHGate(LogicalHGate):

    def __init__(self, concat_code: ConcatenatedCode, ops: TransversalOps,
        prep_strategy: 'PreparationStrategy'=None):
        super().__init__(concat_code)
        self.ops = ops
        self._prep = prep_strategy

    def set_prep(self, prep: 'PreparationStrategy'):
        self._prep = prep

    @property
    def prep(self) ->'PreparationStrategy':
        if self._prep is None:
            raise RuntimeError(
                'Preparation strategy not set for TeleportationHGate')
        return self._prep

    @property
    def implementation_name(self) ->str:
        return 'teleportation'

    def apply(self, circuit: stim.Circuit, loc: int, level: int,
        detector_counter: List[int], ancilla_loc: int=None, p: float=0.0
        ) ->GateResult:
        code = self.concat_code.code_at_level(level)
        N_prev = self.block_size(level - 1) if level > 0 else 1
        N_now = code.n
        N_block = N_prev * N_now
        if ancilla_loc is None:
            ancilla_loc = loc + N_block
        extra_ancilla = [(ancilla_loc + N_block + i * N_block) for i in
            range(code.n * 4)]
        ft_result = self.prep.append_ft_plus_prep(circuit, [ancilla_loc],
            extra_ancilla, N_prev, N_now, p, detector_counter)
        prep_detector_info = ft_result.get('detector_info', [])
        self.ops.append_cz(circuit, loc, ancilla_loc, N_prev, N_now)
        detector_info = self.ops.append_m_x(circuit, loc, N_prev, N_now,
            detector_counter)
        self.ops.append_swap(circuit, loc, ancilla_loc, N_prev, N_now)
        pauli_update = PauliFrameUpdate(x_correction_source=
            'x_measurement_parity', z_correction_source=None,
            source_detectors=detector_info, target_block=loc)
        result = GateResult(self.gate_name, self.implementation_name, level)
        result.detectors = detector_info
        result.pauli_frame_update = pauli_update
        result.metadata = {'ancilla_loc': ancilla_loc, 'swapped_back': True,
            'prep_detector_info': prep_detector_info}
        return result


class TransversalCNOTGate(LogicalCNOTGate):

    def __init__(self, concat_code: ConcatenatedCode, ops: TransversalOps):
        super().__init__(concat_code)
        self.ops = ops

    @property
    def implementation_name(self) ->str:
        return 'transversal'

    def apply(self, circuit, loc_ctrl, loc_targ, level, detector_counter):
        N_prev = self.block_size(level - 1) if level > 0 else 1
        N_now = self.concat_code.code_at_level(level).n
        self.ops.append_cnot(circuit, loc_ctrl, loc_targ, N_prev, N_now)
        return GateResult(self.gate_name, self.implementation_name, level)


class TransversalMeasurement(LogicalMeasurement):

    def __init__(self, concat_code: ConcatenatedCode, ops: TransversalOps):
        super().__init__(concat_code)
        self.ops = ops

    @property
    def implementation_name(self) ->str:
        return 'transversal'

    def apply(self, circuit, loc, level, detector_counter, basis='z'):
        N_prev = self.block_size(level - 1) if level > 0 else 1
        N_now = self.concat_code.code_at_level(level).n
        if basis == 'x':
            self.ops.append_h(circuit, loc, N_prev, N_now)
        detectors = self.ops.append_m(circuit, loc, N_prev, N_now,
            detector_counter)
        result = GateResult(self.gate_name, self.implementation_name, level)
        result.detectors = detectors
        return result


class PreparationStrategy(ABC):

    def __init__(self, concat_code: ConcatenatedCode, ops: TransversalOps):
        self.concat_code = concat_code
        self.ops = ops
        self._ec_gadget = None

    def set_ec_gadget(self, ec_gadget: 'ECGadget'):
        self._ec_gadget = ec_gadget

    @property
    def ec(self) ->'ECGadget':
        if self._ec_gadget is None:
            raise RuntimeError('EC gadget not set')
        return self._ec_gadget

    @property
    @abstractmethod
    def strategy_name(self) ->str:
        pass

    @property
    def uses_prep_ec_at_l2(self) ->bool:
        return True

    @abstractmethod
    def append_0prep(self, circuit: stim.Circuit, loc1: int, N_prev: int,
        N_now: int):
        pass

    @abstractmethod
    def append_verified_0prep(self, circuit: stim.Circuit, loc1: int, loc2:
        int, N_prev: int, N_now: int, p: float, detector_counter: List[int]
        ) ->Union[List, Tuple]:
        pass


class GenericPreparationStrategy(PreparationStrategy):

    def __init__(self, concat_code: ConcatenatedCode, ops: TransversalOps,
        use_idle_noise: bool=True):
        super().__init__(concat_code, ops)
        self.use_idle_noise = use_idle_noise
        self._ft_prep = None

    @property
    def strategy_name(self) ->str:
        return 'generic'

    def _get_ft_prep(self) ->'ShorVerifiedPrepStrategy':
        if self._ft_prep is None:
            self._ft_prep = ShorVerifiedPrepStrategy(self.concat_code, self
                .ops, use_idle_noise=self.use_idle_noise)
        return self._ft_prep

    def append_0prep(self, circuit: stim.Circuit, loc1: int, N_prev: int,
        N_now: int):
        if N_prev == 1:
            code = self.concat_code.code_at_level(0)
        else:
            code = (self.concat_code.code_at_level(1) if self.concat_code.
                num_levels > 1 else self.concat_code.code_at_level(0))
        block_count = _get_code_transversal_block_count(code)
        n_now = block_count if block_count else code.n
        if N_prev == 1:
            PhysicalOps.reset(circuit, loc1, n_now)
            self._measure_stabilizers_for_projection(circuit, loc1, code,
                'zero')
        else:
            for i in range(n_now):
                self.append_0prep(circuit, (loc1 + i) * N_prev, 1, N_prev)

    def _measure_stabilizers_for_projection(self, circuit: stim.Circuit,
        loc1: int, code: 'CSSCode', state_type: str) ->Dict[str, Any]:
        n = code.n
        ancilla = loc1 + n
        result = {'Z_meas_start': circuit.num_measurements, 'Z_meas_end':
            None, 'X_meas_start': None, 'X_meas_end': None, 'Lz_meas_idx':
            None, 'Lx_meas_idx': None}
        hz = code.hz if hasattr(code, 'hz') else code._hz
        for row_idx in range(hz.shape[0]):
            circuit.append('R', [ancilla])
            circuit.append('H', [ancilla])
            for q in range(n):
                if hz[row_idx, q]:
                    circuit.append('CX', [loc1 + q, ancilla])
            circuit.append('H', [ancilla])
            circuit.append('M', [ancilla])
        result['Z_meas_end'] = circuit.num_measurements
        result['X_meas_start'] = circuit.num_measurements
        hx = code.hx if hasattr(code, 'hx') else code._hx
        for row_idx in range(hx.shape[0]):
            circuit.append('R', [ancilla])
            circuit.append('H', [ancilla])
            for q in range(n):
                if hx[row_idx, q]:
                    circuit.append('CZ', [ancilla, loc1 + q])
            circuit.append('H', [ancilla])
            circuit.append('M', [ancilla])
        result['X_meas_end'] = circuit.num_measurements
        if state_type == 'zero':
            lz = code.Lz if hasattr(code, 'Lz') else code._lz
            lz_vec = lz.flatten() if len(lz.shape) > 1 else lz
            circuit.append('R', [ancilla])
            circuit.append('H', [ancilla])
            for q in range(n):
                if lz_vec[q]:
                    circuit.append('CX', [loc1 + q, ancilla])
            circuit.append('H', [ancilla])
            circuit.append('M', [ancilla])
            result['Lz_meas_idx'] = circuit.num_measurements - 1
        elif state_type == 'plus':
            lx = code.Lx if hasattr(code, 'Lx') else code._lx
            lx_vec = lx.flatten() if len(lx.shape) > 1 else lx
            circuit.append('R', [ancilla])
            circuit.append('H', [ancilla])
            for q in range(n):
                if lx_vec[q]:
                    circuit.append('CZ', [ancilla, loc1 + q])
            circuit.append('H', [ancilla])
            circuit.append('M', [ancilla])
            result['Lx_meas_idx'] = circuit.num_measurements - 1
        return result

    def append_plus_prep(self, circuit: stim.Circuit, loc1: int, N_prev:
        int, N_now: int, use_true_plus_state: bool=False):
        if N_prev == 1:
            code = self.concat_code.code_at_level(0)
        else:
            code = (self.concat_code.code_at_level(1) if self.concat_code.
                num_levels > 1 else self.concat_code.code_at_level(0))
        block_count = _get_code_transversal_block_count(code)
        n_now = block_count if block_count else code.n
        if N_prev == 1:
            PhysicalOps.reset(circuit, loc1, n_now)
            for q in range(n_now):
                PhysicalOps.h(circuit, loc1 + q)
            self._measure_stabilizers_for_projection(circuit, loc1, code,
                'plus')
        else:
            # Prepare each inner block in |+_L⟩_inner
            for i in range(n_now):
                self.append_plus_prep(circuit, (loc1 + i) * N_prev, 1, N_prev)
            # NOTE: Outer X stabilizer projection is done in _ft_plus_prep_full
            # after verification, not here, to avoid creating ancillas at
            # arbitrary locations for all verification copies.

    def _measure_outer_x_stabilizers_for_plus_projection(self, circuit: stim.Circuit,
        loc1: int, N_prev: int, n_blocks: int, outer_code: CSSCode):
        """Measure outer X stabilizers to project |+_L⟩^⊗n into |+_L⟩_L2.
        
        For |+_L⟩_L2 preparation:
        - Each inner block is in |+_L⟩_inner
        - This gives |+_L⟩^⊗n at the logical level
        - But |+_L⟩^⊗n is a superposition over all 2^n logical basis states
        - |+_L⟩_L2 should only be superposition over outer codewords
        - Measuring outer X stabilizers projects into the outer code space
        
        For self-dual codes like Steane, Hx = Hz, and we measure the X stabilizers
        by using transversal Z basis measurements on ancilla after CZ gates.
        """
        hx = _get_code_hx(outer_code)
        inner_lx = _get_inner_logical_x(self.concat_code)
        inner_lx_flat = inner_lx.flatten() if len(inner_lx.shape) > 1 else inner_lx
        
        # Use a single ancilla qubit right after the L2 block
        # The L2 block spans from loc1*N_prev to (loc1+n_blocks)*N_prev - 1
        # So we use (loc1 + n_blocks)*N_prev as the ancilla
        ancilla_base = (loc1 + n_blocks) * N_prev
        
        # For each outer X stabilizer
        for stab_idx in range(hx.shape[0]):
            # Get support of this X stabilizer
            support = [q for q in range(outer_code.n) if hx[stab_idx, q] == 1]
            
            # Ancilla for this stabilizer (reuse same qubit by resetting)
            ancilla = ancilla_base
            
            circuit.append('R', [ancilla])
            circuit.append('H', [ancilla])
            
            # Apply CZ between ancilla and the logical X operator of each block in support
            # For Steane inner code, Lx acts on qubits 0,1,2 with X
            # CZ(ancilla, target) with ancilla in |+⟩ measures X on target
            for block_idx in support:
                for phys in range(N_prev):
                    if inner_lx_flat[phys] == 1:
                        target_qubit = (loc1 + block_idx) * N_prev + phys
                        circuit.append('CZ', [ancilla, target_qubit])
            
            circuit.append('H', [ancilla])
            circuit.append('M', [ancilla])
            # The measurement projects into the +1 or -1 eigenspace of the outer X stabilizer
            # For |+_L⟩_L2, we should be in +1 eigenspace, so measurement should be 0

    def append_verified_0prep(self, circuit: stim.Circuit, loc1: int, loc2:
        int, N_prev: int, N_now: int, p: float, detector_counter: List[int]
        ) ->'FTVerificationResult':
        ft_prep = self._get_ft_prep()
        data_locs = [loc1]
        code = self.concat_code.code_at_level(0)
        t = (_get_code_distance(code) - 1) // 2
        num_extra_copies = (t + 1) ** 2 - 1
        extra_ancilla = [(loc2 + i) for i in range(num_extra_copies)]
        return ft_prep.append_ft_0prep(circuit, data_locs, extra_ancilla,
            N_prev, N_now, p, detector_counter)

    def append_ft_bell_prep(self, circuit: stim.Circuit, loc1: int, loc2:
        int, extra_ancilla: List[int], N_prev: int, N_now: int, p: float,
        detector_counter: List[int], num_verification_rounds: Optional[int]
        =None) ->dict:
        ft_prep = self._get_ft_prep()
        return ft_prep.append_ft_bell_prep(circuit, loc1, loc2,
            extra_ancilla, N_prev, N_now, p, detector_counter,
            num_verification_rounds=num_verification_rounds)

    def _compute_gamma(self, p: float) ->float:
        return p / 10

    def _get_idle_qubits(self, code: CSSCode, active_qubits: List[int],
        round_name: str=None) ->List[int]:
        all_qubits = set(range(code.n))
        active_set = set(active_qubits)
        idle = list(all_qubits - active_set)
        if (code.idle_schedule and round_name and round_name in code.
            idle_schedule):
            idle = code.idle_schedule[round_name]
        return sorted(idle)

    def _get_cnot_rounds(self, code: CSSCode) ->List[List[Tuple[int, int]]]:
        encoding_cnot_rounds = getattr(code, 'encoding_cnot_rounds', None)
        if encoding_cnot_rounds:
            return encoding_cnot_rounds
        encoding_cnots = getattr(code, 'encoding_cnots', None)
        if not encoding_cnots:
            return []
        rounds = []
        remaining = list(encoding_cnots)
        while remaining:
            current_round = []
            used_qubits = set()
            still_remaining = []
            for ctrl, targ in remaining:
                if ctrl not in used_qubits and targ not in used_qubits:
                    current_round.append((ctrl, targ))
                    used_qubits.add(ctrl)
                    used_qubits.add(targ)
                else:
                    still_remaining.append((ctrl, targ))
            if current_round:
                rounds.append(current_round)
            remaining = still_remaining
        return rounds

    def _get_verification_schedule(self, code: CSSCode) ->List[List[int]]:
        verif_qubits = getattr(code, 'verification_qubits', [])
        if not verif_qubits:
            lx = _get_code_lx(code)
            verif_qubits = [i for i, v in enumerate(lx) if v == 1]
            h_qubits = getattr(code, 'h_qubits', [])
            if len(verif_qubits) == code.n and h_qubits:
                verif_qubits = h_qubits
            if not verif_qubits:
                verif_qubits = list(range((code.n + 1) // 2))
        return [[vq] for vq in verif_qubits]

    def _get_initial_ec_qubits(self, code: CSSCode) ->List[int]:
        initial_qubits = set(code.h_qubits)
        cnot_rounds = self._get_cnot_rounds(code)
        if cnot_rounds:
            for ctrl, targ in cnot_rounds[0]:
                initial_qubits.add(ctrl)
                initial_qubits.add(targ)
        if code.n <= 4:
            return list(range(code.n))
        result = sorted(initial_qubits)
        min_qubits = max(len(code.h_qubits), code.n // 2)
        if len(result) < min_qubits:
            result = list(range(min_qubits))
        return result


class FaultTolerantPrepMixin:

    @property
    def verification_method(self) ->str:
        return 'none'

    def num_copies_required(self, t: int) ->int:
        return 1

    def provides_r_filter(self, r: int) ->bool:
        return False

    def append_ft_0prep(self, circuit: stim.Circuit, data_locs: List[int],
        ancilla_locs: List[int], N_prev: int, N_now: int, p: float,
        detector_counter: List[int]) ->Dict:
        raise NotImplementedError('Subclass must implement append_ft_0prep')

    def append_ft_plus_prep(self, circuit: stim.Circuit, data_locs: List[
        int], ancilla_locs: List[int], N_prev: int, N_now: int, p: float,
        detector_counter: List[int]) ->Dict:
        raise NotImplementedError('Subclass must implement append_ft_plus_prep'
            )

    def append_ft_bell_prep(self, circuit: stim.Circuit, block1_loc: int,
        block2_loc: int, ancilla_locs: List[int], N_prev: int, N_now: int,
        p: float, detector_counter: List[int], num_verification_rounds:
        Optional[int]=None) ->Dict:
        raise NotImplementedError('Subclass must implement append_ft_bell_prep'
            )


class ShorVerifiedPrepStrategy(GenericPreparationStrategy,
    FaultTolerantPrepMixin):

    def __init__(self, concat_code: ConcatenatedCode, ops: TransversalOps,
        num_syndrome_rounds: int=None, use_idle_noise: bool=True):
        super().__init__(concat_code, ops, use_idle_noise)
        self._num_syndrome_rounds = num_syndrome_rounds

    @property
    def strategy_name(self) ->str:
        return 'shor_verified'

    @property
    def verification_method(self) ->str:
        return 'shor'

    def _get_t(self, code: CSSCode) ->int:
        return (_get_code_distance(code) - 1) // 2

    def num_copies_required(self, t: int) ->int:
        return 1

    def provides_r_filter(self, r: int) ->bool:
        code = self.concat_code.code_at_level(0)
        t = self._get_t(code)
        num_rounds = (self._num_syndrome_rounds if self.
            _num_syndrome_rounds is not None else t + 1)
        return r <= t and num_rounds >= t + 1

    def _prepare_cat_state(self, circuit: stim.Circuit, cat_locs: List[int],
        weight: int, p: float, verify_locs: List[int]=None,
        detector_counter: List[int]=None) ->List[int]:
        if weight < 1:
            return []
        circuit.append('H', cat_locs[0])
        if p > 0:
            circuit.append('DEPOLARIZE1', cat_locs[0], p)
        for i in range(1, weight):
            circuit.append('CNOT', [cat_locs[0], cat_locs[i]])
            if p > 0:
                circuit.append('DEPOLARIZE2', [cat_locs[0], cat_locs[i]], p)
        verification_detectors = []
        if verify_locs is not None and len(verify_locs
            ) >= weight - 1 and detector_counter is not None:
            for i in range(weight - 1):
                check_qubit = verify_locs[i]
                circuit.append('R', check_qubit)
                circuit.append('CNOT', [cat_locs[i], check_qubit])
                if p > 0:
                    circuit.append('DEPOLARIZE2', [cat_locs[i], check_qubit], p
                        )
                circuit.append('CNOT', [cat_locs[i + 1], check_qubit])
                if p > 0:
                    circuit.append('DEPOLARIZE2', [cat_locs[i + 1],
                        check_qubit], p)
                circuit.append('M', check_qubit)
                PhysicalOps.detector(circuit, -1)
                detector_counter[0] += 1
                verification_detectors.append(detector_counter[0] - 1)
        return verification_detectors

    def _measure_stabilizer_with_cat(self, circuit: stim.Circuit, data_loc:
        int, cat_locs: List[int], stabilizer: List[int], stab_type: str,
        N_prev: int, p: float, detector_counter: List[int], verify_cat:
        bool=True) ->Dict:
        weight = len(stabilizer)
        inner_code = self.concat_code.code_at_level(0)
        inner_lz_support, inner_lz_pauli_type = _get_code_lz_info(inner_code)
        inner_lx_support, inner_lx_pauli_type = _get_code_lx_info(inner_code)
        if N_prev == 1:
            actual_cat_size = weight
            inner_support = [0]
        else:
            inner_support = (inner_lx_support if stab_type == 'X' else
                inner_lz_support)
            actual_cat_size = weight * len(inner_support)
        verify_locs = None
        if verify_cat:
            if len(cat_locs) >= 2 * actual_cat_size - 1:
                verify_locs = cat_locs[actual_cat_size:2 * actual_cat_size - 1]
            else:
                import warnings
                warnings.warn(
                    f'Cat state verification requested but insufficient ancillas: need {2 * actual_cat_size - 1} cat_locs for weight-{weight} L2 stabilizer (actual cat size {actual_cat_size}), got {len(cat_locs)}. Cat verification DISABLED - this breaks FT!'
                    , RuntimeWarning)
        verification_detectors = self._prepare_cat_state(circuit, cat_locs[
            :actual_cat_size], actual_cat_size, p, verify_locs=verify_locs if
            verify_cat else None, detector_counter=detector_counter if
            verify_cat else None)
        for i, qubit_idx in enumerate(stabilizer):
            block_base = data_loc + qubit_idx * N_prev
            if stab_type == 'X':
                if N_prev == 1:
                    circuit.append('CZ', [cat_locs[i], block_base])
                    if p > 0:
                        circuit.append('DEPOLARIZE2', [cat_locs[i],
                            block_base], p)
                else:
                    support_size = len(inner_support)
                    for j_idx, j in enumerate(inner_support):
                        cat_idx = i * support_size + j_idx
                        circuit.append('CZ', [cat_locs[cat_idx], block_base +
                            j])
                        if p > 0:
                            circuit.append('DEPOLARIZE2', [cat_locs[cat_idx
                                ], block_base + j], p)
            elif N_prev == 1:
                circuit.append('CZ', [cat_locs[i], block_base])
                if p > 0:
                    circuit.append('DEPOLARIZE2', [cat_locs[i], block_base], p)
            else:
                support_size = len(inner_support)
                for j_idx, j in enumerate(inner_support):
                    cat_idx = i * support_size + j_idx
                    circuit.append('CZ', [cat_locs[cat_idx], block_base + j])
                    if p > 0:
                        circuit.append('DEPOLARIZE2', [cat_locs[cat_idx], 
                            block_base + j], p)
        for i in range(actual_cat_size):
            circuit.append('MX', cat_locs[i])
        syndrome_meas_indices = [stim.target_rec(-actual_cat_size + i) for
            i in range(actual_cat_size)]
        syndrome_detectors = []
        for i in range(actual_cat_size):
            circuit.append('R', cat_locs[i])
        if verify_locs:
            for v in verify_locs:
                circuit.append('R', v)
        return {'syndrome_detectors': syndrome_detectors,
            'verification_detectors': verification_detectors,
            'syndrome_meas_indices': syndrome_meas_indices, 'weight':
            weight, 'actual_cat_size': actual_cat_size, 'stab_type': stab_type}

    def _measure_all_stabilizers(self, circuit: stim.Circuit, data_loc: int,
        cat_locs: List[int], code: CSSCode, N_prev: int, p: float,
        detector_counter: List[int], verify_cat: bool=True, include_logical:
        str=None) ->Dict:
        result = {'X_syndromes': [], 'Z_syndromes': [], 'logical_syndrome':
            [], 'cat_verifications': [], 'Z_meas_indices': [],
            'X_meas_indices': [], 'Lz_meas_indices': [], 'Lx_meas_indices': []}
        hz = _get_code_hz(code)
        hx = _get_code_hx(code)
        for row_idx in range(hx.shape[0]):
            support = [i for i in range(code.n) if hx[row_idx, i] == 1]
            if support:
                det_info = self._measure_stabilizer_with_cat(circuit,
                    data_loc, cat_locs, support, 'Z', N_prev, p,
                    detector_counter, verify_cat=verify_cat)
                result['Z_syndromes'].append(det_info['syndrome_detectors'])
                result['cat_verifications'].extend(det_info[
                    'verification_detectors'])
                result['Z_meas_indices'].append(det_info[
                    'syndrome_meas_indices'])
        if include_logical == 'z':
            lz_support, lz_pauli_type = _get_code_lz_info(code)
            if lz_support:
                det_info = self._measure_stabilizer_with_cat(circuit,
                    data_loc, cat_locs, lz_support, lz_pauli_type, N_prev,
                    p, detector_counter, verify_cat=verify_cat)
                result['logical_syndrome'].append(det_info[
                    'syndrome_detectors'])
                result['cat_verifications'].extend(det_info[
                    'verification_detectors'])
                result['Lz_meas_indices'].append(det_info[
                    'syndrome_meas_indices'])
        for row_idx in range(hz.shape[0]):
            support = [i for i in range(code.n) if hz[row_idx, i] == 1]
            if support:
                det_info = self._measure_stabilizer_with_cat(circuit,
                    data_loc, cat_locs, support, 'X', N_prev, p,
                    detector_counter, verify_cat=verify_cat)
                result['X_syndromes'].append(det_info['syndrome_detectors'])
                result['cat_verifications'].extend(det_info[
                    'verification_detectors'])
                result['X_meas_indices'].append(det_info[
                    'syndrome_meas_indices'])
        if include_logical == 'x':
            lx_support, lx_pauli_type = _get_code_lx_info(code)
            if lx_support:
                det_info = self._measure_stabilizer_with_cat(circuit,
                    data_loc, cat_locs, lx_support, lx_pauli_type, N_prev,
                    p, detector_counter, verify_cat=verify_cat)
                result['logical_syndrome'].append(det_info[
                    'syndrome_detectors'])
                result['cat_verifications'].extend(det_info[
                    'verification_detectors'])
                result['Lx_meas_indices'].append(det_info[
                    'syndrome_meas_indices'])
        return result

    def append_ft_0prep(self, circuit: stim.Circuit, data_locs: List[int],
        ancilla_locs: List[int], N_prev: int, N_now: int, p: float,
        detector_counter: List[int]) ->Dict:
        code = self.concat_code.code_at_level(0)
        t = self._get_t(code)
        num_rounds = (self._num_syndrome_rounds if self.
            _num_syndrome_rounds is not None else t + 1)
        data_loc = data_locs[0]
        self.append_0prep(circuit, data_loc, N_prev, N_now)
        if p > 0:
            physical_base = data_loc * N_prev
            for q in range(N_now):
                circuit.append('DEPOLARIZE1', physical_base + q, p)
        physical_start = data_loc * N_prev
        include_lz = 'z'
        all_syndromes = []
        prev_syndrome_info = None
        for round_idx in range(num_rounds):
            syndrome_info = self._measure_all_stabilizers(circuit,
                physical_start, ancilla_locs, code, N_prev, p,
                detector_counter, include_logical=include_lz)
            all_syndromes.append(syndrome_info)
            if round_idx > 0 and prev_syndrome_info is not None:
                self._create_differential_detectors(circuit, syndrome_info,
                    prev_syndrome_info, detector_counter)
            prev_syndrome_info = syndrome_info
        return {'detector_info': all_syndromes, 'accepted_loc': data_loc,
            'num_copies_used': 1, 'num_syndrome_rounds': num_rounds,
            'verification_outcomes': all_syndromes}

    def _create_differential_detectors(self, circuit: stim.Circuit,
        current_info: Dict, prev_info: Dict, detector_counter: List[int]):
        for curr_meas, prev_meas in zip(current_info.get('Z_meas_indices',
            []), prev_info.get('Z_meas_indices', [])):
            if curr_meas and prev_meas:
                targets = list(curr_meas) + list(prev_meas)
                circuit.append('DETECTOR', targets)
                detector_counter[0] += 1
        for curr_meas, prev_meas in zip(current_info.get('X_meas_indices',
            []), prev_info.get('X_meas_indices', [])):
            if curr_meas and prev_meas:
                targets = list(curr_meas) + list(prev_meas)
                circuit.append('DETECTOR', targets)
                detector_counter[0] += 1
        for curr_meas, prev_meas in zip(current_info.get('Lz_meas_indices',
            []), prev_info.get('Lz_meas_indices', [])):
            if curr_meas and prev_meas:
                targets = list(curr_meas) + list(prev_meas)
                circuit.append('DETECTOR', targets)
                detector_counter[0] += 1

    def append_ft_plus_prep(self, circuit: stim.Circuit, data_locs: List[
        int], ancilla_locs: List[int], N_prev: int, N_now: int, p: float,
        detector_counter: List[int]) ->Dict:
        code = self.concat_code.code_at_level(0)
        t = self._get_t(code)
        num_rounds = (self._num_syndrome_rounds if self.
            _num_syndrome_rounds is not None else t + 1)
        data_loc = data_locs[0]
        physical_start = data_loc * N_prev
        self.append_plus_prep(circuit, data_loc, N_prev, N_now)
        if p > 0:
            physical_base = data_loc * N_prev
            for q in range(N_now):
                circuit.append('DEPOLARIZE1', physical_base + q, p)
        all_syndromes = []
        prev_syndrome_info = None
        for round_idx in range(num_rounds):
            syndrome_info = self._measure_all_stabilizers(circuit,
                physical_start, ancilla_locs, code, N_prev, p,
                detector_counter, include_logical='x')
            all_syndromes.append(syndrome_info)
            if round_idx > 0 and prev_syndrome_info is not None:
                self._create_differential_detectors_plus_prep(circuit,
                    syndrome_info, prev_syndrome_info, detector_counter)
            prev_syndrome_info = syndrome_info
        return {'detector_info': all_syndromes, 'accepted_loc': data_loc,
            'num_copies_used': 1, 'num_syndrome_rounds': num_rounds,
            'verification_outcomes': all_syndromes}

    def _create_differential_detectors_plus_prep(self, circuit: stim.
        Circuit, current_info: Dict, prev_info: Dict, detector_counter:
        List[int]):
        for curr_meas, prev_meas in zip(current_info.get('Z_meas_indices',
            []), prev_info.get('Z_meas_indices', [])):
            if curr_meas and prev_meas:
                targets = list(curr_meas) + list(prev_meas)
                circuit.append('DETECTOR', targets)
                detector_counter[0] += 1
        for curr_meas, prev_meas in zip(current_info.get('X_meas_indices',
            []), prev_info.get('X_meas_indices', [])):
            if curr_meas and prev_meas:
                targets = list(curr_meas) + list(prev_meas)
                circuit.append('DETECTOR', targets)
                detector_counter[0] += 1
        for curr_meas, prev_meas in zip(current_info.get('Lx_meas_indices',
            []), prev_info.get('Lx_meas_indices', [])):
            if curr_meas and prev_meas:
                targets = list(curr_meas) + list(prev_meas)
                circuit.append('DETECTOR', targets)
                detector_counter[0] += 1

    def append_ft_bell_prep(self, circuit: stim.Circuit, block1_loc: int,
        block2_loc: int, ancilla_locs: List[int], N_prev: int, N_now: int,
        p: float, detector_counter: List[int], num_verification_rounds:
        Optional[int]=None) ->Dict:
        code = self.concat_code.code_at_level(0)
        n_ancilla = len(ancilla_locs)
        third = n_ancilla // 3
        result1 = self.append_ft_plus_prep(circuit, [block1_loc],
            ancilla_locs[:third], N_prev, N_now, p, detector_counter)
        result2 = self.append_ft_0prep(circuit, [block2_loc], ancilla_locs[
            third:2 * third], N_prev, N_now, p, detector_counter)
        for i in range(N_now):
            ctrl = block1_loc * N_prev + i
            targ = block2_loc * N_prev + i
            circuit.append('CNOT', [ctrl, targ])
            if p > 0:
                circuit.append('DEPOLARIZE2', [ctrl, targ], p)
        return {'detector_info': {'block1': result1['detector_info'],
            'block2': result2['detector_info']}, 'block1_loc': block1_loc,
            'block2_loc': block2_loc, 'num_copies_used': 2,
            'verification_outcomes': {'block1': result1[
            'verification_outcomes'], 'block2': result2[
            'verification_outcomes']}}


class SteaneVerifiedPrepStrategy(GenericPreparationStrategy,
    FaultTolerantPrepMixin):

    def __init__(self, concat_code: ConcatenatedCode, ops: TransversalOps,
        use_optimized_perfect: bool=True, use_idle_noise: bool=True):
        super().__init__(concat_code, ops, use_idle_noise)
        self.use_optimized_perfect = use_optimized_perfect

    @property
    def strategy_name(self) ->str:
        return 'steane_verified'

    @property
    def verification_method(self) ->str:
        return 'steane'

    def _get_t(self, code: CSSCode) ->int:
        return (_get_code_distance(code) - 1) // 2

    def _is_perfect_code(self, code: CSSCode) ->bool:
        code_d = _get_code_distance(code)
        code_k = _get_code_k(code)
        if code.n == 7 and code_k == 1 and code_d == 3:
            return True
        if code.n == 23 and code_k == 1 and code_d == 7:
            return True
        return False

    def num_copies_required(self, t: int) ->int:
        return (t + 1) ** 2

    def provides_r_filter(self, r: int) ->bool:
        code = self.concat_code.code_at_level(0)
        t = self._get_t(code)
        return r <= t

    def _prepare_multiple_copies(self, circuit: stim.Circuit, locs: List[
        int], N_prev: int, N_now: int, p: float, code: CSSCode) ->None:
        if len(locs) == 0:
            return
        for loc in locs:
            self.append_0prep(circuit, loc, N_prev, N_now)
        if p > 0:
            for loc in locs:
                for q in range(N_now):
                    if N_prev == 1:
                        phys_addr = loc * code.n + q
                    else:
                        phys_addr = loc * N_now + q
                    circuit.append('DEPOLARIZE1', phys_addr, p)

    def _compare_copies_z_basis(self, circuit: stim.Circuit, kept_loc: int,
        sacrificed_loc: int, N_prev: int, code: CSSCode, p: float,
        detector_counter: List[int]) ->List:
        lz_support, lz_pauli_type = _get_code_lz_info(code)
        if lz_pauli_type == 'X':
            return self._compare_copies_for_x_type_lz(circuit, kept_loc,
                sacrificed_loc, N_prev, code, p, detector_counter, lz_support)
        else:
            return self._compare_copies_z_basis_standard(circuit, kept_loc,
                sacrificed_loc, N_prev, code, p, detector_counter, lz_support)

    def _compare_copies_for_x_type_lz(self, circuit: stim.Circuit, kept_loc:
        int, sacrificed_loc: int, N_prev: int, code: CSSCode, p: float,
        detector_counter: List[int], lz_support: List[int]) ->List:
        for q in range(code.n):
            self.ops.append_noisy_cz(circuit, (kept_loc + q) * N_prev, (
                sacrificed_loc + q) * N_prev, 1, N_prev, p)
        if N_prev == 1:
            for q in range(code.n):
                circuit.append('MX', sacrificed_loc + q)
            hx = _get_code_hx(code)
            detector_info = []
            num_stabs = hx.shape[0] if hx is not None else 0
            for s in range(num_stabs):
                support = [i for i in range(code.n) if hx[s, i] == 1]
                meas_refs = [(-(code.n - i)) for i in support]
                circuit.append('DETECTOR', [stim.target_rec(r) for r in
                    meas_refs])
                detector_counter[0] += 1
                detector_info.append(detector_counter[0] - 1)
            if lz_support:
                lz_meas_refs = [(-(code.n - i)) for i in lz_support]
                circuit.append('DETECTOR', [stim.target_rec(r) for r in
                    lz_meas_refs])
                detector_counter[0] += 1
                detector_info.append(detector_counter[0] - 1)
            return detector_info
        else:
            import warnings
            warnings.warn(
                'X-type Lz at L2+ not fully implemented, using standard comparison'
                )
            return self._compare_copies_z_basis_standard(circuit, kept_loc,
                sacrificed_loc, N_prev, code, p, detector_counter, lz_support)

    def _compare_copies_z_basis_standard(self, circuit: stim.Circuit,
        kept_loc: int, sacrificed_loc: int, N_prev: int, code: CSSCode, p:
        float, detector_counter: List[int], lz_support: List[int]) ->List:
        N_now = code.n * N_prev
        for q in range(code.n):
            kept_inner_base = kept_loc * N_now + q * N_prev
            sacrificed_inner_base = sacrificed_loc * N_now + q * N_prev
            self.ops.append_noisy_cnot(circuit, kept_inner_base,
                sacrificed_inner_base, 1, N_prev, p)
        if N_prev == 1:
            sacrificed_base = sacrificed_loc * N_now
            for q in range(code.n):
                circuit.append('M', sacrificed_base + q)
            hz = _get_code_hz(code)
            detector_info = []
            num_stabs = hz.shape[0]
            for s in range(num_stabs):
                support = [i for i in range(code.n) if hz[s, i] == 1]
                meas_refs = [(-(code.n - i)) for i in support]
                circuit.append('DETECTOR', [stim.target_rec(r) for r in
                    meas_refs])
                detector_counter[0] += 1
                detector_info.append(detector_counter[0] - 1)
            lz = _get_code_lz(code)
            lz_support = [i for i in range(code.n) if lz[i] == 1]
            lz_meas_refs = [(-(code.n - i)) for i in lz_support]
            circuit.append('DETECTOR', [stim.target_rec(r) for r in
                lz_meas_refs])
            detector_counter[0] += 1
            detector_info.append(detector_counter[0] - 1)
            return detector_info
        else:
            inner_code = self.concat_code.code_at_level(0)
            inner_lz = _get_code_lz(inner_code)
            N_now_for_sacrificed = code.n * N_prev
            sacrificed_copy_base = sacrificed_loc * N_now_for_sacrificed
            meas_start_per_block = []
            for q in range(code.n):
                meas_start_per_block.append(0)
                inner_base = sacrificed_copy_base + q * N_prev
                for phys in range(N_prev):
                    circuit.append('M', inner_base + phys)
            total_inner_meas = code.n * N_prev
            detector_info = []
            inner_Hz = _get_code_hz(inner_code)
            num_inner_stabs = inner_Hz.shape[0]
            for q in range(code.n):
                for s in range(num_inner_stabs):
                    support = [phys for phys in range(N_prev) if inner_Hz[s,
                        phys] == 1]
                    meas_refs = []
                    for phys in support:
                        meas_idx = -(total_inner_meas - q * N_prev - phys)
                        meas_refs.append(meas_idx)
                    circuit.append('DETECTOR', [stim.target_rec(r) for r in
                        meas_refs])
                    detector_counter[0] += 1
                    detector_info.append(detector_counter[0] - 1)
            hz = _get_code_hz(code)
            num_stabs = hz.shape[0]
            for s in range(num_stabs):
                support = [q for q in range(code.n) if hz[s, q] == 1]
                all_meas_refs = []
                for q in support:
                    for phys in range(N_prev):
                        if inner_lz[phys] == 1:
                            meas_idx = -(total_inner_meas - q * N_prev - phys)
                            all_meas_refs.append(meas_idx)
                circuit.append('DETECTOR', [stim.target_rec(r) for r in
                    all_meas_refs])
                detector_counter[0] += 1
                detector_info.append(detector_counter[0] - 1)
            lz = _get_code_lz(code)
            lz_support = [q for q in range(code.n) if lz[q] == 1]
            all_meas_refs = []
            for q in lz_support:
                for phys in range(N_prev):
                    if inner_lz[phys] == 1:
                        meas_idx = -(total_inner_meas - q * N_prev - phys)
                        all_meas_refs.append(meas_idx)
            circuit.append('DETECTOR', [stim.target_rec(r) for r in
                all_meas_refs])
            detector_counter[0] += 1
            detector_info.append(detector_counter[0] - 1)
            return detector_info

    def _compare_copies_x_basis(self, circuit: stim.Circuit, kept_loc: int,
        sacrificed_loc: int, N_prev: int, code: CSSCode, p: float,
        detector_counter: List[int], check_logical: bool=True) ->List:
        N_now = code.n * N_prev
        for q in range(code.n):
            sacrificed_inner_base = sacrificed_loc * N_now + q * N_prev
            kept_inner_base = kept_loc * N_now + q * N_prev
            self.ops.append_noisy_cnot(circuit, sacrificed_inner_base,
                kept_inner_base, 1, N_prev, p)
        if N_prev == 1:
            sacrificed_base = sacrificed_loc * N_now
            for q in range(code.n):
                circuit.append('MX', sacrificed_base + q)
            hx = _get_code_hx(code)
            detector_info = []
            num_stabs = hx.shape[0]
            for s in range(num_stabs):
                support = [i for i in range(code.n) if hx[s, i] == 1]
                meas_refs = [(-(code.n - i)) for i in support]
                circuit.append('DETECTOR', [stim.target_rec(r) for r in
                    meas_refs])
                detector_counter[0] += 1
                detector_info.append(detector_counter[0] - 1)
            if check_logical:
                lx = _get_code_lx(code)
                lx_support = [i for i in range(code.n) if lx[i] == 1]
                lx_meas_refs = [(-(code.n - i)) for i in lx_support]
                circuit.append('DETECTOR', [stim.target_rec(r) for r in
                    lx_meas_refs])
                detector_counter[0] += 1
                detector_info.append(detector_counter[0] - 1)
            return detector_info
        else:
            inner_code = self.concat_code.code_at_level(0)
            inner_lx = _get_code_lx(inner_code)
            sacrificed_copy_base = sacrificed_loc * N_now
            for q in range(code.n):
                inner_base = sacrificed_copy_base + q * N_prev
                for phys in range(N_prev):
                    circuit.append('MX', inner_base + phys)
            total_inner_meas = code.n * N_prev
            detector_info = []
            inner_Hx = _get_code_hx(inner_code)
            num_inner_stabs = inner_Hx.shape[0]
            for q in range(code.n):
                for s in range(num_inner_stabs):
                    support = [phys for phys in range(N_prev) if inner_Hx[s,
                        phys] == 1]
                    meas_refs = []
                    for phys in support:
                        meas_idx = -(total_inner_meas - q * N_prev - phys)
                        meas_refs.append(meas_idx)
                    circuit.append('DETECTOR', [stim.target_rec(r) for r in
                        meas_refs])
                    detector_counter[0] += 1
                    detector_info.append(detector_counter[0] - 1)
            hx = _get_code_hx(code)
            num_stabs = hx.shape[0]
            for s in range(num_stabs):
                support = [q for q in range(code.n) if hx[s, q] == 1]
                all_meas_refs = []
                for q in support:
                    for phys in range(N_prev):
                        if inner_lx[phys] == 1:
                            meas_idx = -(total_inner_meas - q * N_prev - phys)
                            all_meas_refs.append(meas_idx)
                circuit.append('DETECTOR', [stim.target_rec(r) for r in
                    all_meas_refs])
                detector_counter[0] += 1
                detector_info.append(detector_counter[0] - 1)
            if check_logical:
                lx = _get_code_lx(code)
                lx_support = [q for q in range(code.n) if lx[q] == 1]
                all_meas_refs = []
                for q in lx_support:
                    for phys in range(N_prev):
                        if inner_lx[phys] == 1:
                            meas_idx = -(total_inner_meas - q * N_prev - phys)
                            all_meas_refs.append(meas_idx)
                circuit.append('DETECTOR', [stim.target_rec(r) for r in
                    all_meas_refs])
                detector_counter[0] += 1
                detector_info.append(detector_counter[0] - 1)
            return detector_info

    def append_verified_0prep(self, circuit: stim.Circuit, loc1: int, loc2:
        int, N_prev: int, N_now: int, p: float, detector_counter: List[int]
        ) ->Dict:
        code = self.concat_code.code_at_level(0)
        t = self._get_t(code)
        num_copies = self.num_copies_required(t)
        data_locs = [loc1]
        for i in range(1, num_copies):
            data_locs.append(loc2 + (i - 1) * code.n)
        ancilla_locs = []
        return self.append_ft_0prep(circuit, data_locs, ancilla_locs,
            N_prev, N_now, p, detector_counter)

    def append_ft_0prep(self, circuit: stim.Circuit, data_locs: List[int],
        ancilla_locs: List[int], N_prev: int, N_now: int, p: float,
        detector_counter: List[int]) ->Dict:
        code = self.concat_code.code_at_level(0)
        t = self._get_t(code)
        return self._ft_0prep_full(circuit, data_locs, ancilla_locs, N_prev,
            N_now, p, detector_counter, code, t)

    def _ft_0prep_full(self, circuit: stim.Circuit, data_locs: List[int],
        ancilla_locs: List[int], N_prev: int, N_now: int, p: float,
        detector_counter: List[int], code: CSSCode, t: int) ->Dict:
        group_size = t + 1
        num_groups = t + 1
        total_copies = group_size * num_groups
        all_locs = list(data_locs) + list(ancilla_locs)
        if len(all_locs) < total_copies:
            raise ValueError(
                f'Need {total_copies} locations for copies, got {len(data_locs)} data + {len(ancilla_locs)} ancilla = {len(all_locs)}'
                )
        copy_locs = all_locs[:total_copies]
        self._prepare_multiple_copies(circuit, copy_locs, N_prev, N_now, p,
            code)
        verification_results = {'level1_phase': [], 'level2_bitflip': []}
        phase_verified_locs = []
        for g in range(num_groups):
            group_start = g * group_size
            group_locs = copy_locs[group_start:group_start + group_size]
            kept = group_locs[0]
            for i in range(1, group_size):
                det_info = self._compare_copies_z_basis(circuit, kept,
                    group_locs[i], N_prev, code, p, detector_counter)
                verification_results['level1_phase'].append({'group': g,
                    'kept': kept, 'sacrificed': group_locs[i],
                    'detector_info': det_info})
            phase_verified_locs.append(kept)
        final_kept = phase_verified_locs[0]
        for i in range(1, len(phase_verified_locs)):
            det_info = self._compare_copies_z_basis(circuit, final_kept,
                phase_verified_locs[i], N_prev, code, p, detector_counter)
            verification_results['level2_bitflip'].append({'kept':
                final_kept, 'sacrificed': phase_verified_locs[i],
                'detector_info': det_info})
        return {'detector_info': verification_results, 'accepted_loc':
            final_kept, 'num_copies_used': total_copies,
            'verification_method': 'full_two_level',
            'verification_outcomes': verification_results}

    def append_ft_plus_prep(self, circuit: stim.Circuit, data_locs: List[
        int], ancilla_locs: List[int], N_prev: int, N_now: int, p: float,
        detector_counter: List[int]) ->Dict:
        code = self.concat_code.code_at_level(0)
        t = self._get_t(code)
        return self._ft_plus_prep_full(circuit, data_locs, ancilla_locs,
            N_prev, N_now, p, detector_counter, code, t)

    def _prepare_multiple_plus_copies(self, circuit: stim.Circuit, locs:
        List[int], N_prev: int, N_now: int, p: float, code: CSSCode) ->None:
        for loc in locs:
            self.append_0prep(circuit, loc, N_prev, N_now)
            for q in range(code.n):
                for phys in range(N_prev):
                    circuit.append('H', (loc + q) * N_prev + phys)
            if p > 0:
                for q in range(code.n):
                    for phys in range(N_prev):
                        circuit.append('DEPOLARIZE1', (loc + q) * N_prev +
                            phys, p)

    def _compare_copies_x_basis_for_plus(self, circuit: stim.Circuit,
        kept_loc: int, sacrificed_loc: int, N_prev: int, code: CSSCode, p:
        float, detector_counter: List[int]) ->List:
        lx_support, lx_pauli_type = _get_code_lx_info(code)
        detector_info = []
        if N_prev == 1:
            for q in lx_support:
                circuit.append('CZ', [kept_loc + q, sacrificed_loc + q])
            if p > 0:
                for q in lx_support:
                    circuit.append('DEPOLARIZE2', [kept_loc + q, 
                        sacrificed_loc + q], p)
            for q in range(code.n):
                circuit.append('H', sacrificed_loc + q)
            for q in range(code.n):
                circuit.append('M', sacrificed_loc + q)
            hz = _get_code_hz(code)
            num_stabs = hz.shape[0]
            for s in range(num_stabs):
                support = [q for q in range(code.n) if hz[s, q] == 1]
                meas_refs = [(-(code.n - q)) for q in support]
                circuit.append('DETECTOR', [stim.target_rec(r) for r in
                    meas_refs])
                detector_counter[0] += 1
                detector_info.append(detector_counter[0] - 1)
            lx_meas_refs = [(-(code.n - q)) for q in lx_support]
            circuit.append('DETECTOR', [stim.target_rec(r) for r in
                lx_meas_refs])
            detector_counter[0] += 1
            detector_info.append(detector_counter[0] - 1)
        else:
            inner_lz = _get_inner_logical_z(self.concat_code)
            total_inner_meas = N_prev * code.n
            for q in lx_support:
                for phys in range(N_prev):
                    if inner_lz[phys] == 1:
                        kept_phys = (kept_loc + q) * N_prev + phys
                        sac_phys = (sacrificed_loc + q) * N_prev + phys
                        circuit.append('CZ', [kept_phys, sac_phys])
                        if p > 0:
                            circuit.append('DEPOLARIZE2', [kept_phys,
                                sac_phys], p)
            for q in range(code.n):
                for phys in range(N_prev):
                    circuit.append('H', (sacrificed_loc + q) * N_prev + phys)
            for q in range(code.n):
                for phys in range(N_prev):
                    circuit.append('M', (sacrificed_loc + q) * N_prev + phys)
            hz = _get_code_hz(code)
            num_stabs = hz.shape[0]
            inner_lz_flat = inner_lz.flatten() if len(inner_lz.shape
                ) > 1 else inner_lz
            for s in range(num_stabs):
                support = [q for q in range(code.n) if hz[s, q] == 1]
                all_meas_refs = []
                for q in support:
                    for phys in range(N_prev):
                        if inner_lz_flat[phys] == 1:
                            meas_idx = -(total_inner_meas - q * N_prev - phys)
                            all_meas_refs.append(meas_idx)
                circuit.append('DETECTOR', [stim.target_rec(r) for r in
                    all_meas_refs])
                detector_counter[0] += 1
                detector_info.append(detector_counter[0] - 1)
            all_meas_refs = []
            for q in lx_support:
                for phys in range(N_prev):
                    if inner_lz_flat[phys] == 1:
                        meas_idx = -(total_inner_meas - q * N_prev - phys)
                        all_meas_refs.append(meas_idx)
            circuit.append('DETECTOR', [stim.target_rec(r) for r in
                all_meas_refs])
            detector_counter[0] += 1
            detector_info.append(detector_counter[0] - 1)
        return detector_info

    def _ft_plus_prep_full(self, circuit: stim.Circuit, data_locs: List[int
        ], ancilla_locs: List[int], N_prev: int, N_now: int, p: float,
        detector_counter: List[int], code: CSSCode, t: int) ->Dict:
        group_size = t + 1
        num_groups = t + 1
        total_copies = group_size * num_groups
        if len(data_locs) < total_copies:
            raise ValueError(
                f'Need {total_copies} data locations, got {len(data_locs)}')
        copy_locs = data_locs[:total_copies]
        for loc in copy_locs:
            self.append_plus_prep(circuit, loc, N_prev, N_now)
            if p > 0:
                for q in range(code.n):
                    for phys in range(N_prev):
                        circuit.append('DEPOLARIZE1', (loc + q) * N_prev +
                            phys, p)
        verification_results = {'level1_bitflip': [], 'level2_phase': []}
        bitflip_verified_locs = []
        for g in range(num_groups):
            group_start = g * group_size
            group_locs = copy_locs[group_start:group_start + group_size]
            kept = group_locs[0]
            for i in range(1, group_size):
                det_info = self._compare_copies_z_basis(circuit, kept,
                    group_locs[i], N_prev, code, p, detector_counter)
                verification_results['level1_bitflip'].append({'group': g,
                    'kept': kept, 'sacrificed': group_locs[i],
                    'detector_info': det_info})
            bitflip_verified_locs.append(kept)
        final_kept = bitflip_verified_locs[0]
        for i in range(1, len(bitflip_verified_locs)):
            det_info = self._compare_copies_x_basis_for_plus(circuit,
                final_kept, bitflip_verified_locs[i], N_prev, code, p,
                detector_counter)
            verification_results['level2_phase'].append({'kept': final_kept,
                'sacrificed': bitflip_verified_locs[i], 'detector_info':
                det_info})
        
        # CRITICAL: At L2 (N_prev > 1), must project into outer code space!
        # The verification above checks inner blocks, but |+_L⟩^⊗n is NOT |+_L⟩_L2
        # Need to measure outer X stabilizers to project into the L2 code space
        if N_prev > 1:
            n_blocks = code.n
            self._measure_outer_x_stabilizers_for_plus_projection(
                circuit, final_kept, N_prev, n_blocks, code)
        
        return {'detector_info': verification_results, 'accepted_loc':
            final_kept, 'num_copies_used': total_copies,
            'verification_method': 'full_two_level',
            'verification_outcomes': verification_results}

    def append_ft_bell_prep(self, circuit: stim.Circuit, block1_loc: int,
        block2_loc: int, ancilla_locs: List[int], N_prev: int, N_now: int,
        p: float, detector_counter: List[int], num_verification_rounds:
        Optional[int]=None) ->Dict:
        code = self.concat_code.code_at_level(0)
        t = self._get_t(code)
        code_d = _get_code_distance(code)
        l2_distance = code_d ** 2
        t_l2 = (l2_distance - 1) // 2
        theoretical_rounds = t_l2 + 1
        if num_verification_rounds is not None:
            num_bell_verification_rounds = num_verification_rounds
        else:
            num_bell_verification_rounds = theoretical_rounds
        num_copies = self.num_copies_required(t)
        total_locs_needed = 2 * num_copies
        if len(ancilla_locs) < total_locs_needed:
            raise ValueError(
                f'Steane verification needs {total_locs_needed} ancilla locations, got {len(ancilla_locs)}. For t={t}, need {num_copies} copies per block.'
                )
        plus_copy_locs = [block1_loc] + list(ancilla_locs[:num_copies - 1])
        zero_copy_locs = [block2_loc] + list(ancilla_locs[num_copies - 1:2 *
            num_copies - 2])
        result1 = self.append_ft_plus_prep(circuit, plus_copy_locs, [],
            N_prev, N_now, p, detector_counter)
        result2 = self.append_ft_0prep(circuit, zero_copy_locs, [], N_prev,
            N_now, p, detector_counter)
        num_blocks = N_now // N_prev if N_prev > 1 else N_now
        self.ops.append_noisy_cnot(circuit, block1_loc, block2_loc, N_prev,
            num_blocks, p)
        return {'detector_info': {'plus_prep': result1['detector_info'],
            'zero_prep': result2['detector_info']}, 'block1_loc':
            block1_loc, 'block2_loc': block2_loc, 'num_copies_used': 2 *
            num_copies, 'verification_method': self.verification_method,
            'verification_outcomes': {'plus_prep': result1[
            'verification_outcomes'], 'zero_prep': result2[
            'verification_outcomes']}}


class ECGadget(ABC):

    def __init__(self, concat_code: ConcatenatedCode, ops: TransversalOps):
        self.concat_code = concat_code
        self.ops = ops
        self._prep = None

    def set_prep(self, prep: PreparationStrategy):
        self._prep = prep

    @property
    def prep(self) ->PreparationStrategy:
        if self._prep is None:
            raise RuntimeError('Preparation strategy not set')
        return self._prep

    @property
    @abstractmethod
    def ec_type(self) ->str:
        pass

    @abstractmethod
    def append_noisy_ec(self, circuit: stim.Circuit, loc1: int, loc2: int,
        loc3: int, loc4: int, N_prev: int, N_now: int, p: float,
        detector_counter: List[int]) ->Tuple:
        pass


class KnillECGadget(ECGadget):

    def __init__(self, concat_code: ConcatenatedCode, ops: TransversalOps):
        super().__init__(concat_code, ops)

    @property
    def ec_type(self) ->str:
        return 'knill'

    def _has_ft_prep(self) ->bool:
        return hasattr(self.prep, 'append_ft_bell_prep') and callable(getattr
            (self.prep, 'append_ft_bell_prep', None))

    def append_noisy_ec(self, circuit: stim.Circuit, loc1: int, loc2: int,
        loc3: int, loc4: int, N_prev: int, N_now: int, p: float,
        detector_counter: List[int], measurement_counter: List[int]=None
        ) ->Tuple:
        detector_0prep = []
        detector_0prep_l2 = []
        detector_Z = []
        detector_X = []
        measurement_X = []
        measurement_Z = []
        if N_now == 1:
            return None
        num_blocks = N_now // N_prev if N_prev > 0 else N_now
        if self.concat_code.num_levels > 1:
            outer_code = self.concat_code.code_at_level(1)
            outer_block_count = _get_code_transversal_block_count(outer_code)
            if N_now // N_prev == outer_code.n and outer_block_count:
                num_blocks = outer_block_count
        inner_code = self.concat_code.code_at_level(0)
        t = (_get_code_distance(inner_code) - 1) // 2
        if hasattr(self.prep, 'num_copies_required'):
            num_copies = self.prep.num_copies_required(t)
        else:
            num_copies = (t + 1) ** 2
        inner_lz = _get_code_lz(inner_code)
        lz_support_size = np.sum(inner_lz)
        inner_lx = _get_code_lx(inner_code)
        lx_support_size = np.sum(inner_lx)
        min_ancillas_z = int(2 * lz_support_size + (2 * lz_support_size - 1))
        min_ancillas_x = int(2 * lx_support_size + (2 * lx_support_size - 1))
        bell_verify_total = min_ancillas_z + min_ancillas_x
        bell_verify_requirement = 3 * bell_verify_total * max(N_prev, 1)
        steane_estimate = (2 * num_copies + 2) * max(N_prev, 1)
        shor_estimate = inner_code.n * 2 * (t + 1) * max(N_prev, 1)
        practical_minimum = inner_code.n * 4 * max(N_prev, 1)
        required_ancilla = max(bell_verify_requirement, steane_estimate,
            shor_estimate, practical_minimum)
        if measurement_counter is not None:
            meas_count_before = circuit.num_measurements
        ft_result = self.prep.append_ft_bell_prep(circuit, loc2, loc3, [(
            loc4 * N_prev + i) for i in range(required_ancilla)], N_prev,
            N_now, p, detector_counter)
        if measurement_counter is not None:
            meas_count_after = circuit.num_measurements
            ft_prep_measurements = meas_count_after - meas_count_before
            measurement_counter[0] += ft_prep_measurements
        if 'detector_info' in ft_result:
            if isinstance(ft_result['detector_info'], dict):
                for key, val in ft_result['detector_info'].items():
                    if isinstance(val, list):
                        detector_0prep.extend(val)
            else:
                detector_0prep.extend(ft_result['detector_info'])
        self.ops.append_noisy_cnot(circuit, loc1, loc2, N_prev, num_blocks, p)
        if measurement_counter is not None:
            meas_x_result = self.ops.append_raw_m_x(circuit, loc1, N_prev,
                num_blocks, p, measurement_counter)
            meas_z_result = self.ops.append_raw_m_z(circuit, loc2, N_prev,
                num_blocks, p, measurement_counter)
            detector_X.append(None)
            detector_Z.append(None)
            measurement_X.append(meas_x_result)
            measurement_Z.append(meas_z_result)
        else:
            detector_X.append(self.ops.append_noisy_m_x(circuit, loc1,
                N_prev, num_blocks, p, detector_counter))
            detector_Z.append(self.ops.append_noisy_m_z(circuit, loc2,
                N_prev, num_blocks, p, detector_counter))
        if N_prev == 1:
            if measurement_counter is not None:
                return (detector_0prep, detector_Z, detector_X, loc3,
                    measurement_X, measurement_Z)
            else:
                return detector_0prep, detector_Z, detector_X, loc3
        elif measurement_counter is not None:
            return (detector_0prep, detector_0prep_l2, detector_Z,
                detector_X, loc3, measurement_X, measurement_Z)
        else:
            return (detector_0prep, detector_0prep_l2, detector_Z,
                detector_X, loc3)


class Decoder(ABC):

    def __init__(self, concat_code: ConcatenatedCode):
        self.concat_code = concat_code

    @abstractmethod
    def decode_measurement(self, m: np.ndarray, m_type: str='x') ->int:
        pass


class KnillDecoder(Decoder):

    def __init__(self, concat_code: ConcatenatedCode):
        super().__init__(concat_code)
        self.code = concat_code.code_at_level(0)
        self.n = self.code.n
        self._hz = self.code.Hz
        self._hx = self.code.Hx
        hz = self._hz
        hx = self._hx
        self._logical_x = self.code.Lx
        self._logical_z = self.code.Lz
        self._lz_pauli_type = self.code.lz_pauli_type
        self._lx_pauli_type = self.code.lx_pauli_type
        self._check_matrix_for_lz = hz if self._lz_pauli_type == 'Z' else hx
        self._check_matrix_for_lx = hx if self._lx_pauli_type == 'X' else hz
        self._syndrome_to_qubit_z = self._build_syndrome_lookup(hz)
        self._syndrome_to_qubit_x = self._build_syndrome_lookup(hx)

    def _build_syndrome_lookup(self, check_matrix: np.ndarray) ->Dict[int, int
        ]:
        num_stabilizers = check_matrix.shape[0]
        n = check_matrix.shape[1]
        lookup = {(0): -1}
        for qubit in range(n):
            syndrome = 0
            for stab_idx in range(num_stabilizers):
                if check_matrix[stab_idx, qubit] == 1:
                    syndrome += 1 << stab_idx
            if syndrome not in lookup:
                lookup[syndrome] = qubit
        return lookup

    def _decode_with_syndrome_correction(self, m: np.ndarray, basis: str='z'
        ) ->int:
        m = np.array(m, dtype=int)
        if basis == 'z':
            check_matrix = self._hz
            lookup = self._syndrome_to_qubit_z
            logical_op = self._logical_z
        else:
            check_matrix = self._hx
            lookup = self._syndrome_to_qubit_x
            logical_op = self._logical_x
        syndrome_vec = check_matrix @ m % 2
        syndrome_int = 0
        for i, s in enumerate(syndrome_vec):
            if s:
                syndrome_int += 1 << i
        error_qubit = lookup.get(syndrome_int, -1)
        if error_qubit >= 0:
            m_corrected = m.copy()
            m_corrected[error_qubit] ^= 1
        else:
            m_corrected = m
        return int(np.sum(m_corrected * logical_op) % 2)

    def _compute_error_weights(self, check_matrix: np.ndarray) ->Dict[int, int
        ]:
        num_stabilizers = check_matrix.shape[0]
        n = check_matrix.shape[1]
        weights = {(0): 0}
        for qubit in range(n):
            syndrome = 0
            for stab_idx in range(num_stabilizers):
                if check_matrix[stab_idx, qubit] == 1:
                    syndrome += 1 << stab_idx
            if syndrome not in weights:
                weights[syndrome] = 1
        for q1 in range(n):
            for q2 in range(q1 + 1, n):
                syndrome = 0
                for stab_idx in range(num_stabilizers):
                    bit = (check_matrix[stab_idx, q1] + check_matrix[
                        stab_idx, q2]) % 2
                    if bit == 1:
                        syndrome += 1 << stab_idx
                if syndrome not in weights:
                    weights[syndrome] = 2
        return weights

    def _compute_logical_value(self, m: np.ndarray, logical_op: np.ndarray
        ) ->int:
        return int(np.sum(m * logical_op) % 2)

    def decode_measurement(self, m: np.ndarray, m_type: str='x') ->int:
        if m_type == 'x':
            return self._compute_logical_value(m, self._logical_x)
        else:
            return self._compute_logical_value(m, self._logical_z)

    def decode_final_measurement_l2(self, sample: np.ndarray, detector_m:
        List, pauli_frame: 'PauliFrame', basis: str='z', 
        after_knill_ec: bool=False) ->int:
        """Decode final measurement for L2 concatenated code.
        
        Always applies syndrome correction on final measurements to correct physical errors.
        The Pauli frame tracks logical corrections from Knill EC teleportation.
        """
        n = len(pauli_frame.x_corrections)
        inner_outcomes = np.zeros(n, dtype=int)
        
        for i in range(n):
            if i < len(detector_m):
                det = detector_m[i]
                if isinstance(det, (list, tuple)) and len(det
                    ) >= 2 and isinstance(det[0], int):
                    m_data = np.array(sample[det[0]:det[1]], dtype=int)
                    
                    # Always apply syndrome correction on final measurements
                    raw = self._decode_with_syndrome_correction(m_data, basis)
                    
                    # Apply Pauli frame correction (from Knill EC teleportation)
                    if basis == 'z':
                        correction = pauli_frame.x_corrections[i]
                    else:
                        correction = pauli_frame.z_corrections[i]
                    inner_outcomes[i] = (raw + correction) % 2
                    
        # Outer level decoding with syndrome correction
        outer_raw = self._decode_with_syndrome_correction(inner_outcomes, basis)
            
        if basis == 'z':
            outer_correction = pauli_frame.outer_x
        else:
            outer_correction = pauli_frame.outer_z
        return (outer_raw + outer_correction) % 2

    def decode_ec_l2(self, sample: np.ndarray, ec_result: 'KnillECResult',
        pauli_frame: 'PauliFrame') ->'PauliFrame':
        n = len(pauli_frame.x_corrections)
        outer_meas_z = ec_result.measurement_Z[-1
            ] if ec_result.measurement_Z else None
        outer_meas_x = ec_result.measurement_X[-1
            ] if ec_result.measurement_X else None
        if outer_meas_z is None:
            outer_meas_z = ec_result.detector_Z[-1
                ] if ec_result.detector_Z else None
        if outer_meas_x is None:
            outer_meas_x = ec_result.detector_X[-1
                ] if ec_result.detector_X else None
        inner_lz_values = np.zeros(n, dtype=int)
        inner_lx_values = np.zeros(n, dtype=int)
        if outer_meas_z is not None and isinstance(outer_meas_z, list):
            if len(outer_meas_z) > 0 and isinstance(outer_meas_z[0], list):
                for block_idx in range(min(n, len(outer_meas_z))):
                    meas_z = outer_meas_z[block_idx]
                    if isinstance(meas_z, list) and len(meas_z) >= 2:
                        m_data = np.array(sample[meas_z[0]:meas_z[1]],
                            dtype=int)
                        inner_lz_values[block_idx
                            ] = self._compute_logical_value(m_data, self.
                            _logical_z)
            elif len(outer_meas_z) >= 2 and isinstance(outer_meas_z[0], int):
                # L1 case: single block, MZ result → Z correction for X measurement
                m_data = np.array(sample[outer_meas_z[0]:outer_meas_z[1]],
                    dtype=int)
                z_meas = self._compute_logical_value(m_data, self._logical_z)
                if z_meas == 1:
                    pauli_frame.apply_outer_z(1)  # Z correction from MZ result
        if outer_meas_x is not None and isinstance(outer_meas_x, list):
            if len(outer_meas_x) > 0 and isinstance(outer_meas_x[0], list):
                for block_idx in range(min(n, len(outer_meas_x))):
                    meas_x = outer_meas_x[block_idx]
                    if isinstance(meas_x, list) and len(meas_x) >= 2:
                        m_data = np.array(sample[meas_x[0]:meas_x[1]],
                            dtype=int)
                        inner_lx_values[block_idx
                            ] = self._compute_logical_value(m_data, self.
                            _logical_x)
            elif len(outer_meas_x) >= 2 and isinstance(outer_meas_x[0], int):
                # L1 case: single block, MX result → X correction for Z measurement
                m_data = np.array(sample[outer_meas_x[0]:outer_meas_x[1]],
                    dtype=int)
                x_meas = self._compute_logical_value(m_data, self._logical_x)
                if x_meas == 1:
                    pauli_frame.apply_outer_x(1)  # X correction from MX result
        for block_idx in range(n):
            # MX result (inner_lx) determines X correction needed for Z-basis measurements
            # After teleportation: state is Z^mz X^mx |data⟩
            # For Z measurement: X^mx flips the result, so we need x_correction from mx
            if inner_lx_values[block_idx] == 1:
                pauli_frame.apply_x_correction(block_idx, 1)
            # MZ result (inner_lz) determines Z correction needed for X-basis measurements
            # For X measurement: Z^mz flips the result, so we need z_correction from mz
            if inner_lz_values[block_idx] == 1:
                pauli_frame.apply_z_correction(block_idx, 1)
        return pauli_frame


class PostSelector:

    def __init__(self, concat_code: ConcatenatedCode, decoder: Decoder):
        self.concat_code = concat_code
        self.decoder = decoder

    def post_selection_prep_detectors(self, x: np.ndarray, detector_0prep: List
        ) ->bool:
        if not detector_0prep:
            return True
        for det in detector_0prep:
            if det is None:
                continue
            if isinstance(det, int):
                if x[det] != 0:
                    return False
            elif isinstance(det, (list, tuple)):
                if len(det) == 2 and isinstance(det[0], int) and isinstance(det
                    [1], int):
                    if any(x[det[0]:det[1]]):
                        return False
                elif not self.post_selection_prep_detectors(x, det):
                    return False
        return True


class MemoryAcceptanceChecker:

    def __init__(self, concat_code: ConcatenatedCode, decoder: 'KnillDecoder'):
        self.concat_code = concat_code
        self.decoder = decoder
        self.k = concat_code.code_at_level(0
            ).k if concat_code.num_levels > 0 else 1

    def check_l2(self, sample: np.ndarray, detector_m: List, pauli_frame:
        'PauliFrame', basis: str='z', after_knill_ec: bool=False) ->bool:
        outcome = self.decoder.decode_final_measurement_l2(sample,
            detector_m, pauli_frame, basis, after_knill_ec=after_knill_ec)
        return outcome == 0

    def count_errors_l2(self, sample: np.ndarray, detector_m: List,
        pauli_frame: 'PauliFrame', basis: str='z', after_knill_ec: bool=False) ->float:
        return 0.0 if self.check_l2(sample, detector_m, pauli_frame, basis,
            after_knill_ec=after_knill_ec) else 1.0


AcceptanceChecker = MemoryAcceptanceChecker


class ConcatenatedMemoryExperiment(Experiment):

    def __init__(self, concat_code: ConcatenatedCode, noise_model:
        NoiseModel=None, num_ec_rounds: int=1, ec_gadget: 'ECGadget'=None,
        prep_strategy: 'PreparationStrategy'=None, decoder: 'Decoder'=None,
        metadata: Optional[Dict[str, Any]]=None, post_selection_threshold:
        Optional[int]=0):
        super().__init__(code=concat_code, noise_model=noise_model,
            metadata=metadata or {})
        self.concat_code = concat_code
        self.num_ec_rounds = num_ec_rounds
        self.ops = TransversalOps(concat_code)
        if ec_gadget is not None:
            self.ec = ec_gadget
        else:
            self.ec = KnillECGadget(concat_code, self.ops)
        if prep_strategy is not None:
            self.prep = prep_strategy
        else:
            self.prep = ShorVerifiedPrepStrategy(concat_code, self.ops)
        if decoder is not None:
            self.decoder_knill = decoder
        else:
            self.decoder_knill = KnillDecoder(concat_code)
        self.ec.set_prep(self.prep)
        self.prep.set_ec_gadget(self.ec)
        self.post_selector = PostSelector(concat_code, self.decoder_knill)
        self.memory_acceptance = MemoryAcceptanceChecker(concat_code, self.
            decoder_knill)
        self.post_selection_threshold = post_selection_threshold
        self._circuit_metadata = {}

    def to_stim(self) ->stim.Circuit:
        inner_code = self.concat_code.code_at_level(0)
        N_prev = inner_code.n
        N_now = inner_code.n * inner_code.n
        n = inner_code.n
        NN = 2 * n
        list_ec_results = []
        list_verification_detectors = []
        circuit = stim.Circuit()
        detector_counter = [0]
        measurement_counter = [0]
        uses_shor_prep = isinstance(self.prep, ShorVerifiedPrepStrategy)
        uses_steane_prep = isinstance(self.prep, SteaneVerifiedPrepStrategy)

        def extract_cat_verification_detectors(obj):
            detectors = []
            if isinstance(obj, dict):
                if 'cat_verifications' in obj:
                    cat_dets = obj['cat_verifications']
                    if isinstance(cat_dets, list):
                        detectors.extend(cat_dets)
                    elif isinstance(cat_dets, int):
                        detectors.append(cat_dets)
                for key in ['detector_info', 'l2_detector_info']:
                    if key in obj:
                        detectors.extend(extract_cat_verification_detectors
                            (obj[key]))
            return detectors

        def extract_steane_verification_detectors(obj):
            detectors = []
            if obj is None:
                return detectors
            if isinstance(obj, dict):
                if 'detector_info' in obj:
                    det_info = obj['detector_info']
                    if isinstance(det_info, list):
                        for item in det_info:
                            if isinstance(item, int):
                                detectors.append(item)
                    elif isinstance(det_info, int):
                        detectors.append(det_info)
                    elif isinstance(det_info, dict):
                        detectors.extend(extract_steane_verification_detectors
                            (det_info))
                for key in ['level1_phase', 'level2_bitflip',
                    'level1_bitflip', 'level2_phase', 'z_comparisons',
                    'x_comparisons', 'verification_outcomes']:
                    if key in obj:
                        val = obj[key]
                        if isinstance(val, list):
                            for item in val:
                                detectors.extend(
                                    extract_steane_verification_detectors(item)
                                    )
                        elif isinstance(val, dict):
                            detectors.extend(
                                extract_steane_verification_detectors(val))
            return detectors
        p = 0.0
        num_meas_before_prep = circuit.num_measurements
        prep_result = self.prep.append_verified_0prep(circuit, 0, NN,
            N_prev, N_now, p, detector_counter)
        num_meas_after_prep = circuit.num_measurements
        measurement_counter[0] = num_meas_after_prep
        if prep_result is not None:
            if uses_shor_prep:
                list_verification_detectors.extend(
                    extract_cat_verification_detectors(prep_result))
            elif uses_steane_prep:
                list_verification_detectors.extend(
                    extract_steane_verification_detectors(prep_result))
        locations = [0, NN, 2 * NN, 3 * NN]
        data_loc = 0
        for ec_round_idx in range(self.num_ec_rounds):
            available = [loc for loc in locations if loc != data_loc]
            anc1_loc, anc2_loc, workspace_loc = available[0], available[1
                ], available[2]
            result = self.ec.append_noisy_ec(circuit, data_loc, anc1_loc,
                anc2_loc, workspace_loc, N_prev, N_now, p, detector_counter,
                measurement_counter)
            data_loc = anc2_loc
            if isinstance(result, dict):
                if uses_shor_prep:
                    list_verification_detectors.extend(
                        extract_cat_verification_detectors(result.get(
                        'detector_info', [])))
                    list_verification_detectors.extend(
                        extract_cat_verification_detectors(result))
                elif uses_steane_prep:
                    list_verification_detectors.extend(
                        extract_steane_verification_detectors(result))
            elif isinstance(result, tuple) and len(result) >= 4:
                prep_det = result[0]
                prep_det_l2 = result[1] if len(result) > 1 else None
                if uses_shor_prep:
                    list_verification_detectors.extend(
                        extract_cat_verification_detectors(prep_det))
                    if prep_det_l2:
                        list_verification_detectors.extend(
                            extract_cat_verification_detectors(prep_det_l2))
                elif uses_steane_prep:
                    list_verification_detectors.extend(
                        extract_steane_verification_detectors(prep_det))
                    if prep_det_l2:
                        list_verification_detectors.extend(
                            extract_steane_verification_detectors(prep_det_l2))
            ec_result = KnillECResult.from_tuple_l2(result)
            list_ec_results.append(ec_result)
        lz_pauli_type = self.decoder_knill._lz_pauli_type
        final_meas_start = measurement_counter[0]
        if lz_pauli_type == 'X':
            meas_m = self.ops.append_raw_m_x(circuit, data_loc, N_prev, n, 
                0.0, measurement_counter)
            measurement_basis = 'x'
        else:
            meas_m = self.ops.append_raw_m_z(circuit, data_loc, N_prev, n, 
                0.0, measurement_counter)
            measurement_basis = 'z'
        final_meas_end = measurement_counter[0]
        self._circuit_metadata = {'list_ec_results': list_ec_results,
            'list_verification_detectors': list_verification_detectors,
            'final_meas_range': [final_meas_start, final_meas_end],
            'meas_m': meas_m, 'N_prev': N_prev, 'N_now': N_now, 'n': n, 'k':
            inner_code.k, 'measurement_basis': measurement_basis,
            'prep_meas_range': [num_meas_before_prep, num_meas_after_prep]}
        return circuit

    def run_decode(self, shots: int=10000, decoder_name: Optional[str]=None
        ) ->Dict[str, Any]:
        import random
        circuit = self.to_stim()
        if self.noise_model is not None:
            circuit = self.noise_model.apply(circuit)
        meta = self._circuit_metadata
        list_ec_results = meta['list_ec_results']
        list_verification_detectors = meta['list_verification_detectors']
        meas_m = meta['meas_m']
        n = meta['n']
        k = meta['k']
        measurement_basis = meta.get('measurement_basis', 'z')
        shared_seed = random.randint(0, 2 ** 31 - 1)
        det_samples = circuit.compile_detector_sampler(seed=shared_seed
            ).sample(shots=shots)
        meas_samples = circuit.compile_sampler(seed=shared_seed).sample(shots
            =shots)
        verification_detectors = list_verification_detectors
        if self.post_selection_threshold is None:
            accepted_indices = list(range(len(meas_samples)))
        elif self.post_selection_threshold == 0:
            accepted_indices = [i for i, x in enumerate(det_samples) if
                self.post_selector.post_selection_prep_detectors(x,
                verification_detectors)]
        else:

            def count_firings(det_sample, detectors):
                return sum(1 for d in detectors if d < len(det_sample) and
                    det_sample[d])
            accepted_indices = [i for i, x in enumerate(det_samples) if 
                count_firings(x, verification_detectors) <= self.
                post_selection_threshold]
        samples = [meas_samples[i] for i in accepted_indices]
        num_accepted = len(samples)
        num_errors = 0
        logical_errors = []
        # Track whether Knill EC rounds were processed
        had_knill_ec = len(list_ec_results) > 0
        for sample_idx, x in enumerate(samples):
            pauli_frame = PauliFrame.for_l2(n=n, k=k)
            for ec_result in list_ec_results:
                pauli_frame = self.decoder_knill.decode_ec_l2(x, ec_result,
                    pauli_frame)
            error_count = self.memory_acceptance.count_errors_l2(x, meas_m,
                pauli_frame, basis=measurement_basis, after_knill_ec=had_knill_ec)
            num_errors += error_count
            logical_errors.append(1 if error_count > 0 else 0)
        if num_accepted > 0:
            logical_error_rate = num_errors / (num_accepted * k)
        else:
            logical_error_rate = 0.0
        return {'shots': shots, 'accepted': num_accepted, 'logical_errors':
            np.array(logical_errors, dtype=np.uint8), 'logical_error_rate':
            float(logical_error_rate), 'num_errors': num_errors}


def create_memory_experiment(concat_code: ConcatenatedCode, noise_model:
    NoiseModel=None, num_ec_rounds: int=1) ->ConcatenatedMemoryExperiment:
    ops = TransversalOps(concat_code)
    prep = ShorVerifiedPrepStrategy(concat_code, ops)
    return ConcatenatedMemoryExperiment(concat_code=concat_code,
        noise_model=noise_model, num_ec_rounds=num_ec_rounds, prep_strategy
        =prep)


if __name__ == '__main__':
    import sys
    import json
    print('For Steane code simulation, use concatenated_css_v10_steane.py')
    print(
        'This module (concatenated_css_v10.py) contains only generic CSS code infrastructure.'
        )
    print()
    print('Example usage:')
    print(
        '  from concatenated_css_v10_steane import create_concatenated_steane')
    print('  from concatenated_css_v10 import create_memory_experiment')
    print('  from qectostim.noise.models import CircuitDepolarizingNoise')
    print('  ')
    print('  code = create_concatenated_steane(num_levels=2)')
    print('  noise = CircuitDepolarizingNoise(p1=0.001, p2=0.001)')
    print('  exp = create_memory_experiment(code, noise, num_ec_rounds=1)')
    print('  results = exp.run_decode(shots=10000)')
    print('  print(f\'Logical error rate: {results["logical_error_rate"]}\')')
