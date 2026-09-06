"""Demonstrate physical bit packing of quantized codes."""

import numpy as np

from minitorch import Tensor, explain_packing, pack_codes, unpack_codes


def main() -> None:
    codes = np.array([10, 3], dtype=np.int32)
    packed = pack_codes(codes, bits=4)

    print("MiniTorch Bit Packing Demo")
    print("==========================")
    print(explain_packing(codes, bits=4))
    print()
    print(f"packed byte: 0x{int(packed[0]):02X}")
    print(f"unpacked: {unpack_codes(packed, bits=4, count=2)}")
    print()

    data = np.linspace(0.0, 1.0, 1000)
    x = Tensor(data)
    logical = x.quantize(bits=4)
    packed_q = x.quantize(bits=4, packed=True)

    logical_report = logical.report()
    packed_report = packed_q.report()

    print("QuantizedTensor storage comparison (4-bit uniform, n=1000)")
    print(
        f"Logical int32 bytes: {logical_report.memory.logical_int32_storage_bytes}"
    )
    print(f"Actual packed bytes:   {packed_report.memory.actual_packed_bytes}")
    print(f"Theoretical packed:    {packed_report.memory.theoretical_packed_bytes}")
    print(f"RMSE logical:          {logical_report.error.rmse:.6f}")
    print(f"RMSE packed:           {packed_report.error.rmse:.6f}")


if __name__ == "__main__":
    main()
