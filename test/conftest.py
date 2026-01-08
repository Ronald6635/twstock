import requests

# Keep original
_orig_get = requests.get

class DummyResp:
    def __init__(self, obj):
        self._obj = obj

    def json(self):
        return self._obj


def _make_twse_data(capacity_list, close_list):
    data = []
    # Dates in ROC year 104 => 2015
    for i, (cap, close) in enumerate(zip(capacity_list, close_list), start=1):
        day = ((i - 1) % 28) + 1
        date = f"104/05/{day:02d}"
        # Build row: date, capacity, turnover, open, high, low, close, change, transaction
        row = [date, str(cap), str(0), f"{close}", f"{close}", f"{close}", f"{close}", "0.0", "1000"]
        data.append(row)
    return data


def _make_tpex_aaData(capacity_list, close_list):
    aa = []
    for i, (cap, close) in enumerate(zip(capacity_list, close_list), start=1):
        day = ((i - 1) % 28) + 1
        date = f"104/05/{day:02d}"
        # TPEX uses strings where capacity and turnover are in thousands
        row = [date, str(int(cap/1000)), str(int(0)), f"{close}", f"{close}", f"{close}", f"{close}", "0.0", "1000"]
        aa.append(row)
    return aa

# Expected lists from tests
_TWSE_CAP = [
    30868640,27789400,18824208,21908150,20035646,20402529,24956498,19437537,39888654,24831890,
    26212375,26321396,26984912,41286686,22103852,16323218,16069726,24257941,36704395,61983862
]
_TWSE_CLOSE = [147.5,147.0,147.5,146.5,146.5,148.5,147.5,148.0,146.0,146.5,146.5,146.5,146.5,145.5,145.5,147.5,146.5,145.0,147.0,146.0]

_TPEX_CAP = [
    374000,474000,468000,1257000,1079000,3400000,3424000,1078000,1433000,891000,
    1202000,1008000,999000,488000,706000,231000,1890000,782000,1214000,583000
]
_TPEX_CLOSE = [91.4,91.8,91.8,93.5,91.0,84.7,84.0,85.8,87.1,86.1,83.9,84.5,86.7,86.3,86.0,86.2,91.1,90.9,91.7,90.4]

TWSE_SAMPLE = {"stat": "OK", "data": _make_twse_data(_TWSE_CAP, _TWSE_CLOSE)}
TPEX_SAMPLE = {"aaData": _make_tpex_aaData(_TPEX_CAP, _TPEX_CLOSE)}


def fake_get(url, params=None, **kwargs):
    import urllib.parse
    import copy
    # Realtime stock info endpoint
    if "getStockInfo.jsp" in url:
        q = urllib.parse.urlparse(url).query
        qs = urllib.parse.parse_qs(q)
        ex_ch = qs.get('ex_ch', [''])[0]
        if not ex_ch or '9999' in ex_ch:
            return DummyResp({'rtcode': '5001', 'rtmessage': 'Empty Query.'})
        # Build aggregated response from mock
        import twstock as _twstock
        items = ex_ch.split('|')
        # Single known stock -> return full mock if available
        if len(items) == 1 and '2330' in items[0]:
            return DummyResp(_twstock.mock.get_stock_info('2330'))
        if len(items) == 1 and '6223' in items[0]:
            # mock doesn't have 6223; synthesize from 2330 sample
            base = _twstock.mock.get_stock_info('2330').copy()
            base['msgArray'] = [dict(base['msgArray'][0])]
            base['msgArray'][0]['c'] = '6223'
            base['msgArray'][0]['ch'] = '6223.tw'
            base['msgArray'][0]['n'] = '\u66c7\u77f3'  # placeholder name
            return DummyResp(base)
        # Multiple stocks: aggregate msgArray
        agg = {'msgArray': []}
        for item in items:
            if '2330' in item:
                agg['msgArray'].extend(_twstock.mock.get_stock_info('2330')['msgArray'])
            if '6223' in item:
                # synth from 2330
                base = _twstock.mock.get_stock_info('2330')
                entry = dict(base['msgArray'][0])
                entry['c'] = '6223'
                entry['ch'] = '6223.tw'
                entry['n'] = '\u66c7\u77f3'
                agg['msgArray'].append(entry)
        if not agg['msgArray']:
            return DummyResp({'rtcode': '5001', 'rtmessage': 'Empty Query.'})
        agg['rtcode'] = '0000'
        agg['rtmessage'] = 'OK'
        return DummyResp(agg)

    # TWSE STOCK_DAY
    if "exchangeReport/STOCK_DAY" in url and params is not None:
        stockNo = params.get("stockNo") or params.get("stockNo")
        date = params.get("date", "")
        # Specific test for 2015 May
        if stockNo == "2330" and date.startswith("201505"):
            return DummyResp(copy.deepcopy(TWSE_SAMPLE))
        # Generic: return repeated sample for recent months to support fetch_31
        if stockNo == "2330":
            big = {"stat": "OK", "data": _make_twse_data(_TWSE_CAP * 4, _TWSE_CLOSE * 4)}
            return DummyResp(copy.deepcopy(big))
    # TPEX daily
    if "st43_result.php" in url and params is not None:
        if params.get("stkno") == "6223" and params.get("d") == "104/5":
            return DummyResp(copy.deepcopy(TPEX_SAMPLE))
        if params.get("stkno") == "6223":
            big = {"aaData": _make_tpex_aaData(_TPEX_CAP * 4, _TPEX_CLOSE * 4)}
            return DummyResp(copy.deepcopy(big))
    return _orig_get(url, params=params, **kwargs)

import pytest

# Monkeypatch for tests: apply fake_get during the test session and restore originals afterwards
@pytest.fixture(autouse=True, scope='session')
def _patch_requests():
    orig_get = requests.get
    orig_session_get = requests.Session.get

    requests.get = fake_get
    requests.Session.get = lambda self, url, params=None, **kwargs: fake_get(url, params=params, **kwargs)

    yield

    # restore originals
    requests.get = orig_get
    requests.Session.get = orig_session_get
