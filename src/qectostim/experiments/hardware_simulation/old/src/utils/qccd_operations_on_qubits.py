from src.utils.qccd_operations import *



class QubitOperation(Operation):
    def __init__(
        self,
        run: Callable[[Trap], None],
        ions: Sequence[Ion],
        involvedComponents: Sequence[QCCDComponent],
        **kwargs,
    ) -> None:
        super().__init__(run, involvedComponents, ions=ions, **kwargs)
        self._ions = ions
        self._trap: Optional[Trap] = None

    def _generateLabelAddOns(self) -> None:
        self._involvedIonsForLabel = self._ions
        self._addOns = ""
        for ion in self._ions:
            self._addOns += f" {ion.label}"

    def getTrapForIons(self) -> Optional[Trap]:
        """
        return trap if all ions are in the same trap
        """
        if not self.ions:
            return None
        trap = self.ions[0].parent
        if trap is None:
            return None
        if not isinstance(trap, Trap):
            return None
        if not all(i.parent == trap for i in self.ions[1:]):
            return None
        return trap

    def setTrap(self, trap: Trap) -> None:
        self._kwargs["trap"] = trap
        self._involvedComponents.append(trap)
        self._trap = trap

    @property
    def isApplicable(self) -> bool:
        if not self._trap:
            return False
        for ion in self.ions:
            if ion.parent != self._trap:
                return False 
        return super().isApplicable
    
    def _checkApplicability(self) -> None:
        if not self._trap:
            raise ValueError("QubitOperation: trap not set")
        for ion in self.ions:
            if ion.parent != self._trap:
                raise ValueError(f"QubitOperation: ion {ion.idx} not in trap {self._trap.idx}") 
        return super()._checkApplicability()

    @classmethod
    def physicalOperation(cls, trap: Trap, **kwargs) -> "QubitOperation":
        qubitOperation = cls.qubitOperation(**kwargs)
        qubitOperation.setTrap(trap)
        return qubitOperation

    @classmethod
    @abc.abstractmethod
    def qubitOperation(cls) -> "QubitOperation": ...

    @property
    def ions(self) -> Sequence[Ion]:
        return self._ions

    def run(self) -> None:
        self._checkApplicability()
        self.calculateOperationTime()
        self.calculateFidelity()
        self._run(self._trap)
        self._generateLabelAddOns()


class OneQubitGate(QubitOperation):
    # TODO split into X, Y rotations
    KEY = Operations.ONE_QUBIT_GATE
    GATE_DURATION = (
        5e-6  # TABLE IV https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330
    )
    A = 0.003680029  # Scaling factor for fidelity calculation

    def __init__(
        self,
        run: Callable[[Trap], None],
        ion: Ion,
        involvedComponents: Sequence["QCCDComponent"],
        **kwargs,
    ) -> None:
        super().__init__(
            run, involvedComponents=involvedComponents, ions=[ion], **kwargs
        )

    def calculateOperationTime(self) -> None:
        self._operationTime = self.GATE_DURATION

    def calculateFidelity(self) -> None:
        if self._trap is None:
            raise ValueError("fidelity: trap has not been set")
        self.calculateOperationTime()
        n = (
            (len(self._trap.ions)) / np.log(len(self._trap.ions))
            if len(self._trap.ions) >= 2
            else 1
        )
        # Page 7 https://arxiv.org/pdf/2004.04706
        self._fidelity = (
            1
            - (self._trap.backgroundHeatingRate * self.operationTime()
            + self.A * n * (2 * self._trap.motionalMode + 1))
        )
        # self._fidelity = 1-3e-3

    def calculateDephasingFidelity(self) -> None:
        self._dephasingFidelity = 1 # FIXME might be inaccurate

    @classmethod
    def qubitOperation(cls, ion: Ion) -> "OneQubitGate":
        def run(trap: Trap):
            ...

        return cls(run=run, ion=ion, involvedComponents=[ion])

class XRotation(OneQubitGate):
    @property
    def label(self) -> str:
        return "RX" + self._addOns

class YRotation(OneQubitGate):
    @property
    def label(self) -> str:
        return "RY" + self._addOns

class Measurement(QubitOperation):
    KEY = Operations.MEASUREMENT
    MEASUREMENT_TIME = (
        400e-6  # TABLE IV https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330
    )
    INFIDELITY = (
        1e-3  # TABLE IV https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330
    )

    def __init__(
        self,
        run: Callable[[Trap], bool],
        ion: Ion,
        involvedComponents: Sequence["QCCDComponent"],
        **kwargs,
    ) -> None:
        super().__init__(
            run, involvedComponents=involvedComponents, ions=[ion], **kwargs
        )

    def calculateOperationTime(self) -> None:
        self._operationTime = self.MEASUREMENT_TIME

    def calculateFidelity(self) -> None:
        self._fidelity = 1 - self.INFIDELITY 

    def calculateDephasingFidelity(self) -> None:
        return 1 # we can schedule measurements at the end of the quantum circuit

    @classmethod
    def qubitOperation(cls, ion: Ion) -> "Measurement":
        def run(trap: Trap):
            ...

        return cls(run=run, ion=ion, involvedComponents=[ion])



class QubitReset(QubitOperation):
    KEY = Operations.QUBIT_RESET
    RESET_TIME = (
        50e-6  # TABLE IV https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330
    )
    INFIDELITY = (
        5e-3  # TABLE IV https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330
    )

    def __init__(
        self,
        run: Callable[[Trap], bool],
        ion: Ion,
        involvedComponents: Sequence["QCCDComponent"],
        **kwargs,
    ) -> None:
        super().__init__(
            run, involvedComponents=involvedComponents, ions=[ion], **kwargs
        )

    def calculateOperationTime(self) -> None:
        self._operationTime = self.RESET_TIME

    def calculateFidelity(self) -> None:
        self._fidelity = 1 - self.INFIDELITY 

    def calculateDephasingFidelity(self) -> None:
        return 1 # FIXME determine dephasing due to reset

    @classmethod
    def qubitOperation(cls, ion: Ion) -> "QubitReset":
        def run(trap: Trap):
            ...

        return cls(run=run, ion=ion, involvedComponents=[ion])

class TwoQubitMSGate(QubitOperation):
    KEY = Operations.TWO_QUBIT_MS_GATE
    A = 0.003680029  # Scaling factor for fidelity calculation
    T2 =2.2
    OP_TIME = 40e-6#+850e-6 # Remember to change WISE values

    def __init__(
        self,
        run: Callable[[Trap], bool],
        involvedComponents: Sequence["QCCDComponent"],
        ion1: Ion,
        ion2: Ion,
        gate_type: str = "AM2",
        **kwargs,
    ) -> None:
        super().__init__(
            run, involvedComponents=involvedComponents, **kwargs, ions=[ion1, ion2]
        )
        self.gateType = gate_type
        self._ion1 = ion1
        self._ion2 = ion2

    def calculateOperationTime(self) -> None:
        if self._trap is None:
            raise ValueError("operationTime: trap has not been set")
        # Page 6 https://arxiv.org/pdf/2004.04706
        # distance = np.sqrt((self._ion1.idx - self._ion2.idx) ** 2)
        # chainLength = len(self._trap.ions)

        # if self.gateType == "AM1":
        #     self._operationTime = max(100 * distance - 22, 0) * 1e-6
        # elif self.gateType == "AM2":
        #     self._operationTime = max(38 * distance + 10, 0) * 1e-6
        # elif self.gateType == "PM":
        #     self._operationTime =  max(5 * distance + 160, 0) * 1e-6
        # elif self.gateType == "FM":
        #     self._operationTime = max(13.33 * chainLength - 54, 100) * 1e-6
        self._operationTime = self.OP_TIME

    def calculateFidelity(self) -> None:
        if self._trap is None:
            raise ValueError("fidelity: trap has not been set")
        self.calculateOperationTime()
        n = (
            (len(self._trap.ions)) / np.log(len(self._trap.ions))
            if len(self._trap.ions) >= 2
            else 1
        )
        # Page 7 https://arxiv.org/pdf/2004.04706
        self._fidelity = (
            1
            - (self._trap.backgroundHeatingRate * self.operationTime()
            + self.A * n * (2 * self._trap.motionalMode + 1))
        )

        # self._fidelity = 1-2e-3

    def calculateDephasingFidelity(self) -> None:
        self.calculateOperationTime()
        self._dephasingFidelity = 1 - (1-np.exp(-self.operationTime()/self.T2))/2  # Dephasing noise https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330


    @property
    def ionsActedIdxs(self) -> Tuple[int, int]:
        return self._ion1.idx, self._ion2.idx

    @property
    def ionsInJunctions(self) -> bool:
        return isinstance(self._ion1.parent, Junction) or isinstance(
            self._ion2.parent, Junction
        )

    @classmethod
    def qubitOperation(
        cls, ion1: Ion, ion2: Ion, gate_type: str = "AM2"
    ) -> "TwoQubitMSGate":
        def run(trap: Trap):
            ...

        return cls(
            run=run,
            involvedComponents=[ion1, ion2],
            gate_type=gate_type,
            ion1=ion1,
            ion2=ion2,
        )
    

class GateSwap(QubitOperation):
    KEY = Operations.GATE_SWAP

    def __init__(
        self,
        run: Callable[[Any], bool],
        operations: Sequence[TwoQubitMSGate],
        ion1: Ion,
        ion2: Ion,
        involvedComponents: Sequence["QCCDComponent"],
        **kwargs,
    ) -> None:
        super().__init__(run, involvedComponents=involvedComponents, ions=[ion1,ion2], **kwargs)
        self._operations: Sequence[TwoQubitMSGate] = operations
        self._ion1 = ion1
        self._ion2 = ion2

    def calculateOperationTime(self) -> None:
        for op in self._operations:
            op.calculateOperationTime()
        self._operationTime = sum(op.operationTime() for op in self._operations)

    def calculateDephasingFidelity(self) -> None:
        for op in self._operations:
            op.calculateDephasingFidelity()
        self._dephasingFidelity = np.prod([op.dephasingFidelity() for op in self._operations])

    def calculateFidelity(self) -> None:
        for op in self._operations:
            op.calculateFidelity()
        self._fidelity = np.prod([op.fidelity() for op in self._operations])

    def _generateLabelAddOns(self) -> None:
        self._involvedIonsForLabel = [self._ion1, self._ion2]
        self._addOns = f" {self._ion1.label} {self._ion2.label}"

    def setTrap(self, trap: Trap) -> None:
        for op in self._operations:
            op.setTrap(trap)
        return super().setTrap(trap)

    @classmethod
    def qubitOperation(cls, ion1: Ion, ion2: Ion):
        operations: List[TwoQubitMSGate] = []
        for _ in range(3): # REF: fig. 5 https://arxiv.org/pdf/2004.04706
            operations.append(TwoQubitMSGate.qubitOperation(ion1=ion1, ion2=ion2))

        def run(trap: Trap):
            if ion1 == ion2:
                return
            for op in operations:
                op.run()
            idx1 = trap.ions.index(ion1)
            trap.removeIon(ion1)
            trap.addIon(ion1, adjacentIon=ion2)
            trap.removeIon(ion2)
            trap.addIon(ion2, offset=idx1)

        return cls(
            run=run,
            operations=operations,
            ion1=ion1,
            ion2=ion2,
            involvedComponents=[ion1, ion2],
        )