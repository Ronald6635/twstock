# -*- coding: utf-8 -*-

import twstock


def run(argv):
    def fmt_last5(seq):
        vals = [v if v is not None else '' for v in seq[-5:]]
        # Pad to 5 elements if fewer
        vals = ([''] * max(0, 5 - len(vals))) + vals
        return ' '.join(f"{v:>5}" if v != '' else '    ' for v in vals)

    for sid in argv:
        s = twstock.Stock(sid)
        print("-------------- %s ---------------- " % sid)
        print("high : %s" % fmt_last5(s.high))
        print("low  : %s" % fmt_last5(s.low))
        print("price: %s" % fmt_last5(s.price))
