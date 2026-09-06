"""Canonical operation name constants used across MiniTorch."""

ADD = "add"
SUB = "sub"
MUL = "mul"
DIV = "div"
NEG = "neg"
MATMUL = "matmul"
POW = "pow"
SUM = "sum"
MEAN = "mean"
RESHAPE = "reshape"
TRANSPOSE = "transpose"

ALL_OPERATION_NAMES: tuple[str, ...] = (
    ADD,
    SUB,
    MUL,
    DIV,
    NEG,
    MATMUL,
    POW,
    SUM,
    MEAN,
    RESHAPE,
    TRANSPOSE,
)
