from typing import List, Tuple

import numpy as np
import cirq
from cirq import optimizers


def matrix_to_sycamore_operations(
        target_qubits: List[cirq.GridQubit], matrix: np.ndarray
) -> Tuple[cirq.OP_TREE, List[cirq.GridQubit]]:
    """A method to convert a unitary matrix to a list of Sycamore operations.

    This method will return a list of `cirq.Operation`s using the qubits and (optionally) ancilla
    qubits to implement the unitary matrix `matrix` on the target qubits `qubits`.
    The operations are also supported by `cirq.google.gate_sets.SYC_GATESET`.

    Args:
        target_qubits: list of qubits the returned operations will act on. The qubit order defined by the list
            is assumed to be used by the operations to implement `matrix`.
        matrix: a matrix that is guaranteed to be unitary and of size (2**len(qs), 2**len(qs)).
    Returns:
        A tuple of operations and ancilla qubits allocated.
            Operations: In case the matrix is supported, a list of operations `ops` is returned.
                `ops` acts on `qs` qubits and for which `cirq.unitary(ops)` is equal to `matrix` up
                 to certain tolerance. In case the matrix is not supported, it might return NotImplemented to
                 reduce the noise in the judge output.
            Ancilla qubits: In case ancilla qubits are allocated a list of ancilla qubits. Otherwise
                an empty list.
        .
    """
    num_of_qubits = len(target_qubits)

    if np.all(np.equal(matrix, np.eye(2 ** num_of_qubits))):
        # Simple Identity Check
        ops_list = []
        for qubit in target_qubits:
            op = cirq.Z(qubit) ** 0
            cirq.google.Sycamore.validate_operation(op)
            ops_list.append(cirq.Z(qubit) ** 0)
        return ops_list, []

    if num_of_qubits == 1:
        # single qubit gates
        gate = optimizers.single_qubit_matrix_to_phxz(matrix)
        cirq.google.Sycamore.validate_operation(gate(target_qubits[0]))
        return [gate(target_qubits[0])], []

    elif num_of_qubits == 2:
        # two qubit gates
        ops_list = optimizers.two_qubit_matrix_to_operations(target_qubits[0], target_qubits[1], matrix,
                                                             allow_partial_czs=True)
        ConvertToSycamoreGates = cirq.google.ConvertToSycamoreGates()
        converted_ops_list = ConvertToSycamoreGates.convert(op=ops_list)
        return converted_ops_list, []

    elif is_incremental(matrix):
        ancilla = find_neighbor_available_qubit(target_qubits)
        return decompose_incrementer_matrix(target_qubits, ancilla), [ancilla]

    elif num_of_qubits == 3:
        # three qubit gates
        ops_list = optimizers.three_qubit_matrix_to_operations(target_qubits[0], target_qubits[1], target_qubits[2],
                                                               matrix)

        return ops_list, []

    elif np.count_nonzero(matrix - np.diag(np.diagonal(matrix))) == 0:
        # diagonal gates with more than 3 qubits
        angle_list = []
        for i in np.arange(np.shape(matrix)[0]):
            angle_list.append(np.angle(matrix[i, i]))
        diagonal_gate = cirq.ops.DiagonalGate(angle_list)
        ops_list = diagonal_gate._decompose_(qubits=target_qubits)

        return ops_list, []

    elif num_of_qubits == 4:
        ancilla = find_neighbor_available_qubit(target_qubits)
        CTOFFLI_mat = cirq.unitary(
            cirq.ops.ControlledOperation(controls=[target_qubits[0], target_qubits[1], target_qubits[2]],
                                         sub_operation=cirq.X(target_qubits[3])))
        if np.all(np.equal(matrix, CTOFFLI_mat)):
            ops_list = []
            ConvertToSycamoreGates = cirq.google.ConvertToSycamoreGates()
            decomposed_ops = cirq.optimizers.decompose_multi_controlled_x(
                controls=[target_qubits[2], target_qubits[1], target_qubits[0]], target=target_qubits[3],
                free_qubits=[ancilla])
            ops_list.append(ConvertToSycamoreGates.convert(decomposed_ops))
            return ops_list, [ancilla]


    else:
        ops_list = []
        for qubit in target_qubits:
            op = cirq.Z(qubit) ** 0
            cirq.google.Sycamore.validate_operation(op)
            ops_list.append(cirq.Z(qubit) ** 0)
        return ops_list, []

def is_incremental(mat):
    mat_new = np.zeros([np.shape(mat)[0], np.shape(mat)[1]])
    mat_new[0:-1, :] = mat[1:, :]
    mat_new[-1, :] = mat[0, :]

    return np.count_nonzero(mat_new - np.diag(np.diagonal(mat_new))) == 0


def decompose_incrementer_matrix(target_qubits, ancilla):
    ConvertToSycamoreGates = cirq.google.ConvertToSycamoreGates()
    num_of_qubits = len(target_qubits)
    # assume num_of_qubits>=3
    q = target_qubits[::-1]
    op_list = []

    if num_of_qubits > 7:
        decomposed_ops = cirq.optimizers.decompose_multi_controlled_x(
            controls=[q[0], q[1], q[2], q[3], q[4], q[5], q[6]],
            target=q[7], free_qubits=[ancilla])
        op_list.append(ConvertToSycamoreGates.convert(decomposed_ops))

    if num_of_qubits > 6:
        decomposed_ops = cirq.optimizers.decompose_multi_controlled_x(controls=[q[0], q[1], q[2], q[3], q[4], q[5]],
                                                                      target=q[6], free_qubits=[ancilla])
        op_list.append(ConvertToSycamoreGates.convert(decomposed_ops))

    if num_of_qubits > 5:
        decomposed_ops = cirq.optimizers.decompose_multi_controlled_x(controls=[q[0], q[1], q[2], q[3], q[4]],
                                                                      target=q[5],
                                                                      free_qubits=[ancilla])
        op_list.append(ConvertToSycamoreGates.convert(decomposed_ops))

    if num_of_qubits > 4:
        decomposed_ops = cirq.optimizers.decompose_multi_controlled_x(controls=[q[0], q[1], q[2], q[3]], target=q[4],
                                                                      free_qubits=[ancilla])
        op_list.append(ConvertToSycamoreGates.convert(decomposed_ops))

    if num_of_qubits > 3:
        decomposed_ops = cirq.optimizers.decompose_multi_controlled_x(controls=[q[0], q[1], q[2]], target=q[3],
                                                                      free_qubits=[ancilla])
        op_list.append(ConvertToSycamoreGates.convert(decomposed_ops))
    op_list.append(ConvertToSycamoreGates.convert(op=cirq.TOFFOLI(q[0], q[1], q[2])))

    op_list.append(ConvertToSycamoreGates.convert(op=cirq.CNOT(q[0], q[1])))

    op_list.append(cirq.X(q[0]))

    return op_list

def find_neighbor_available_qubit(target_qubits):
    for target_qubit in target_qubits:
        neighbor_qubits = target_qubit.neighbors()
        for neighbor_qubit in neighbor_qubits:
            if neighbor_qubit not in target_qubits:
                return neighbor_qubit

