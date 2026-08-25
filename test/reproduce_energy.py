#!/usr/bin/env python3
"""Recalculate the energy values reported in TTFSFormer Table 2."""

from decimal import Decimal


E_AC_PJ = Decimal("0.9")
E_MAC_PJ = Decimal("4.6")

# OPSNN (GOP) and ESNN/EANN are transcribed from paper Table 2.
ROWS = (
    ("ViT-S/16", Decimal("5.42"), Decimal("0.223")),
    ("ViT-B/16", Decimal("19.2"), Decimal("0.210")),
    ("ViT-L/16", Decimal("65.7"), Decimal("0.207")),
)


def main() -> None:
    print("model,op_snn_gop,e_snn_mj,reported_ratio,implied_e_ann_mj,implied_op_ann_gop")
    for model, op_snn_gop, ratio in ROWS:
        # GOP * pJ = mJ because 10^9 * 10^-12 J = 10^-3 J.
        e_snn_mj = op_snn_gop * E_AC_PJ
        implied_e_ann_mj = e_snn_mj / ratio
        implied_op_ann_gop = implied_e_ann_mj / E_MAC_PJ
        print(
            f"{model},{op_snn_gop},{e_snn_mj},{ratio},"
            f"{implied_e_ann_mj:.6f},{implied_op_ann_gop:.6f}"
        )


if __name__ == "__main__":
    main()
