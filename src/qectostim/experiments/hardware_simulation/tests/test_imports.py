# src/qectostim/experiments/hardware_simulation/tests/test_imports.py
"""
Test imports for hardware simulation framework.

Verifies that all framework components can be imported successfully.
"""
import pytest


class TestCoreImports:
    """Test core module imports."""
    
    def test_architecture_imports(self):
        """Test architecture module imports."""
        from qectostim.experiments.hardware_simulation.core.architecture import (
            HardwareArchitecture,
            ConnectivityGraph,
            Zone,
            ZoneType,
            PhysicalConstraints,
        )
        assert HardwareArchitecture is not None
        assert ConnectivityGraph is not None
    
    def test_gates_imports(self):
        """Test gates module imports."""
        from qectostim.experiments.hardware_simulation.core.gates import (
            GateSpec,
            GateType,
            NativeGateSet,
            GateDecomposition,
            TRAPPED_ION_NATIVE_GATES,
            SUPERCONDUCTING_NATIVE_GATES,
            NEUTRAL_ATOM_NATIVE_GATES,
        )
        assert GateSpec is not None
        assert len(TRAPPED_ION_NATIVE_GATES) > 0
    
    def test_components_imports(self):
        """Test components module imports."""
        from qectostim.experiments.hardware_simulation.core.components import (
            HardwareComponent,
            PhysicalQubit,
            Coupler,
        )
        assert HardwareComponent is not None
    
    def test_operations_imports(self):
        """Test operations module imports."""
        from qectostim.experiments.hardware_simulation.core.operations import (
            PhysicalOperation,
            TransportOperation,
            GateOperation,
            MeasurementOperation,
        )
        assert PhysicalOperation is not None
    
    def test_pipeline_imports(self):
        """Test pipeline module imports."""
        from qectostim.experiments.hardware_simulation.core.pipeline import (
            NativeCircuit,
            MappedCircuit,
            RoutedCircuit,
            ScheduledCircuit,
            CompiledCircuit,
        )
        assert NativeCircuit is not None
    
    def test_compiler_imports(self):
        """Test compiler module imports."""
        from qectostim.experiments.hardware_simulation.core.compiler import (
            HardwareCompiler,
            CompilationPass,
            CompilationConfig,
        )
        assert HardwareCompiler is not None
        assert CompilationConfig is not None


class TestTrappedIonImports:
    """Test trapped ion platform imports."""
    
    def test_architecture_imports(self):
        """Test trapped ion architecture imports."""
        from qectostim.experiments.hardware_simulation.trapped_ion.architecture import (
            TrappedIonArchitecture,
            QCCDArchitecture,
            LinearChainArchitecture,
        )
        assert TrappedIonArchitecture is not None
    
    def test_compiler_imports(self):
        """Test trapped ion compiler imports."""
        from qectostim.experiments.hardware_simulation.trapped_ion.compiler import (
            TrappedIonCompiler,
            QCCDCompiler,
            LinearChainCompiler,
        )
        assert TrappedIonCompiler is not None
    
    def test_simulator_imports(self):
        """Test trapped ion simulator imports."""
        from qectostim.experiments.hardware_simulation.trapped_ion.simulator import (
            TrappedIonSimulator,
        )
        assert TrappedIonSimulator is not None
    
    def test_noise_imports(self):
        """Test trapped ion noise imports."""
        from qectostim.experiments.hardware_simulation.trapped_ion.noise import (
            TrappedIonNoiseModel,
        )
        assert TrappedIonNoiseModel is not None


class TestSuperconductingImports:
    """Test superconducting platform imports."""
    
    def test_architecture_imports(self):
        """Test superconducting architecture imports."""
        from qectostim.experiments.hardware_simulation.superconducting.architecture import (
            SuperconductingArchitecture,
            FixedCouplingArchitecture,
            TunableCouplerArchitecture,
        )
        assert SuperconductingArchitecture is not None
    
    def test_compiler_imports(self):
        """Test superconducting compiler imports."""
        from qectostim.experiments.hardware_simulation.superconducting.compiler import (
            SuperconductingCompiler,
        )
        assert SuperconductingCompiler is not None
    
    def test_simulator_imports(self):
        """Test superconducting simulator imports."""
        from qectostim.experiments.hardware_simulation.superconducting.simulator import (
            SuperconductingSimulator,
        )
        assert SuperconductingSimulator is not None
    
    def test_noise_imports(self):
        """Test superconducting noise imports."""
        from qectostim.experiments.hardware_simulation.superconducting.noise import (
            SuperconductingNoiseModel,
        )
        assert SuperconductingNoiseModel is not None


class TestNeutralAtomImports:
    """Test neutral atom platform imports."""
    
    def test_architecture_imports(self):
        """Test neutral atom architecture imports."""
        from qectostim.experiments.hardware_simulation.neutral_atom.architecture import (
            NeutralAtomArchitecture,
            TweezerArrayArchitecture,
            RydbergLatticeArchitecture,
        )
        assert NeutralAtomArchitecture is not None
    
    def test_compiler_imports(self):
        """Test neutral atom compiler imports."""
        from qectostim.experiments.hardware_simulation.neutral_atom.compiler import (
            NeutralAtomCompiler,
            TweezerArrayCompiler,
            RydbergLatticeCompiler,
        )
        assert NeutralAtomCompiler is not None
    
    def test_simulator_imports(self):
        """Test neutral atom simulator imports."""
        from qectostim.experiments.hardware_simulation.neutral_atom.simulator import (
            NeutralAtomSimulator,
        )
        assert NeutralAtomSimulator is not None
    
    def test_noise_imports(self):
        """Test neutral atom noise imports."""
        from qectostim.experiments.hardware_simulation.neutral_atom.noise import (
            NeutralAtomNoiseModel,
        )
        assert NeutralAtomNoiseModel is not None


class TestTopLevelImports:
    """Test top-level package imports."""
    
    def test_main_package_imports(self):
        """Test main hardware_simulation package imports."""
        from qectostim.experiments.hardware_simulation import (
            # Core
            HardwareArchitecture,
            HardwareCompiler,
            HardwareSimulator,
            # Trapped Ion
            TrappedIonArchitecture,
            TrappedIonSimulator,
            # Superconducting
            SuperconductingArchitecture,
            SuperconductingSimulator,
            # Neutral Atom
            NeutralAtomArchitecture,
            NeutralAtomSimulator,
            # Registry
            get_platform_registry,
            list_platforms,
        )
        assert HardwareArchitecture is not None
        assert get_platform_registry is not None
    
    def test_platform_registry(self):
        """Test platform registry functions."""
        from qectostim.experiments.hardware_simulation import (
            get_platform_registry,
            get_platform,
            list_platforms,
            list_architectures,
        )
        
        platforms = list_platforms()
        assert "trapped_ion" in platforms
        assert "superconducting" in platforms
        assert "neutral_atom" in platforms
        
        registry = get_platform_registry()
        assert len(registry) == 3
        
        ti_info = get_platform("trapped_ion")
        assert ti_info.name == "trapped_ion"


class TestNoiseModelImports:
    """Test noise model imports."""
    
    def test_base_noise_imports(self):
        """Test base hardware noise imports."""
        from qectostim.noise.hardware import (
            HardwareNoiseModel,
            CalibrationData,
            OperationNoiseModel,
            IdleNoiseModel,
        )
        assert HardwareNoiseModel is not None
        assert CalibrationData is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
