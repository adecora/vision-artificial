def latex_matrix(matrix):
    """
    Función auxiliar para imprimir una matrix en LaTeX.

    Parámetros:
    ===========
    matrix: Matriz a imprimir en LaTeX

    Returns:
    ========
    str: Representación en LaTeX de la matriz
    """
    rows = [" & ".join(f"{x:.4f}" for x in row) for row in matrix]
    return r"\begin{bmatrix}" + r" \\ ".join(rows) + r"\end{bmatrix}"


def latex_vector(vector):
    """
    Función auxiliar para imprimir un vector en LaTeX.

    Parámetros:
    ===========
    vector: Vector a imprimir en LaTeX

    Returns:
    ========
    str: Representación en LaTeX del vector
    """
    return r"\left(" + ", ".join(f"{x:.4f}" for x in vector) + r"\right)"
