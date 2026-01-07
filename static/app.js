const select = document.getElementById('api-select');
const stockIdInput = document.getElementById('stock-id');
const fetchBtn = document.getElementById('fetch-btn');
const autoRefreshCheckbox = document.getElementById('auto-refresh');
const display = document.getElementById('data-display');

let autoRefreshInterval;

function showPopup(message, type = 'success') {
    Toastify({
        text: message,
        duration: 3000,
        gravity: "top",
        position: "right",
        style: {
            background: type === 'error' ? "linear-gradient(to right, #ff5f6d, #ffc371)" : "linear-gradient(to right, #00b09b, #96c93d)",
        },
    }).showToast();
}

async function fetchData(api, stockId) {
    let url;
    if (api === 'finmind') {
        url = `/api/finmind/${stockId}`;
    } else if (api.startsWith('finmind_')) {
        const subApi = api.replace('finmind_', '');
        url = `/api/finmind/${subApi}/${stockId}`;
    } else {
        url = `/api/${api}`;
        if (api !== 'codes') {
            url += `/${stockId}`;
        }
    }

    try {
        const res = await fetch(url);
        const data = await res.json();
        if (data.error) {
            showPopup('錯誤: ' + data.error, 'error');
            display.innerHTML = '<p>資料載入錯誤。</p>';
        } else {
            await displayData(data, api);
            showPopup('資料更新成功');
        }
    } catch (err) {
        showPopup('網路錯誤: ' + err.message, 'error');
        display.innerHTML = '<p>網路錯誤。</p>';
    }
}

async function displayData(data, api) {
    console.log('displayData called with api:', api); // Debug log
    console.log('data structure:', typeof data, Array.isArray(data) ? 'array' : 'object'); // Debug log
    let html = '<h2>資料結果</h2>';
    
    // Define API titles in Traditional Chinese
    const titleMap = {
        'finmind': '台灣股價資料表',
        'finmind_institutional': '個股三大法人買賣表',
        'finmind_margin': '個股融資融券表',
        'finmind_per': '個股 PER、PBR 資料表',
        'finmind_revenue': '個股營收資料表',
        'finmind_weekly': '台股週線',
        'finmind_monthly': '台股月線'
    };
    
    // Update title if available
    if (titleMap[api]) {
        html = `<h2>${titleMap[api]}</h2>`;
    }
    
    // Define dataset names for FinMind APIs (only those supported by translation API)
    const datasetMap = {
        'finmind': 'TaiwanStockPrice',
        'finmind_institutional': 'TaiwanStockInstitutionalInvestorsBuySell',
        'finmind_margin': 'TaiwanStockMarginPurchaseShortSale'
    };
    
    // Define fallback translations for datasets not supported by translation API
    const fallbackTranslations = {
        'finmind': {
            'date': '日期',
            'stock_id': '股票代碼',
            'Trading_Volume': '成交量',
            'Trading_money': '成交金額',
            'open': '開盤價',
            'max': '最高價',
            'min': '最低價',
            'close': '收盤價',
            'spread': '漲跌',
            'Trading_turnover': '成交筆數'
        },
        'finmind_per': {
            'date': '日期',
            'stock_id': '股票代碼',
            'dividend_yield': '股利殖利率',
            'PER': '本益比',
            'PBR': '股價淨值比'
        },
        'finmind_revenue': {
            'date': '日期',
            'stock_id': '股票代碼',
            'country': '國家',
            'revenue': '營收'
        },
        'finmind_weekly': {
            'date': '日期',
            'stock_id': '股票代碼',
            'Trading_Volume': '成交量',
            'Trading_money': '成交金額',
            'open': '開盤價',
            'max': '最高價',
            'min': '最低價',
            'close': '收盤價',
            'spread': '漲跌',
            'Trading_turnover': '成交筆數'
        },
        'finmind_monthly': {
            'date': '日期',
            'stock_id': '股票代碼',
            'Trading_Volume': '成交量',
            'Trading_money': '成交金額',
            'open': '開盤價',
            'max': '最高價',
            'min': '最低價',
            'close': '收盤價',
            'spread': '漲跌',
            'Trading_turnover': '成交筆數'
        }
    };
    
    // Fetch translations if this is a supported FinMind API, otherwise use fallback
    let translations = {};
    if (datasetMap[api]) {
        try {
            const response = await fetch(`/api/finmind/translation/${datasetMap[api]}`);
            if (response.ok) {
                const data = await response.json();
                if (data.data) {
                    translations = data.data;
                }
                console.log('Fetched translations:', translations); // Debug log
            }
        } catch (error) {
            console.log('Failed to fetch translations:', error);
        }
    }
    
    // If no translations fetched, use fallback
    if (!translations || Object.keys(translations).length === 0) {
        translations = fallbackTranslations[api] || {};
        console.log('Using fallback translations:', translations); // Debug log
    }
    
    if (api === 'stock') {
        html += `
            <p>股票代碼: ${data.stock_id}</p>
            <p>目前價格: ${data.current_price}</p>
            <p>開盤: ${data.open_price}, 最高: ${data.high_price}, 最低: ${data.low_price}</p>
            <p>成交量: ${data.volume}</p>
            <p>漲跌幅: ${data.price_change}%</p>
            <p>四大買賣點: ${data.best_four_point}</p>
            <img src="data:image/png;base64,${data.chart_image}" alt="股票圖表">
        `;
    } else if (api === 'realtime') {
        const infoMap = {
            'name': '名稱',
            'time': '時間',
            'code': '代碼'
        };
        const realtimeMap = {
            'latest_trade_price': '最新成交價',
            'trade_volume': '成交量',
            'accumulate_trade_volume': '累計成交量',
            'open': '開盤價',
            'high': '最高價',
            'low': '最低價',
            'best_bid_price': '最佳買價',
            'best_bid_volume': '最佳買量',
            'best_ask_price': '最佳賣價',
            'best_ask_volume': '最佳賣量'
        };
        html += '<h3>股票資訊</h3>';
        if (data.info) {
            html += '<table border="1"><tbody>';
            for (let [key, value] of Object.entries(data.info)) {
                const displayKey = infoMap[key] || key;
                html += `<tr><td>${displayKey}</td><td>${value}</td></tr>`;
            }
            html += '</tbody></table>';
        }
        html += '<h3>即時資料</h3>';
        if (data.realtime) {
            html += '<table border="1"><tbody>';
            for (let [key, value] of Object.entries(data.realtime)) {
                const displayKey = realtimeMap[key] || key;
                html += `<tr><td>${displayKey}</td><td>${value}</td></tr>`;
            }
            html += '</tbody></table>';
        }
    } else if (api === 'analytics') {
        html += `
            <p>股票代碼: ${data.stock_id}</p>
            <p>四大買賣點: ${data.best_four_point}</p>
        `;
    } else if (api === 'codes') {
        html += '<table border="1"><thead><tr><th>股票代碼</th><th>名稱</th><th>類型</th><th>ISIN</th></tr></thead><tbody>';
        for (let [id, info] of Object.entries(data)) {
            html += `<tr><td>${id}</td><td>${info.name}</td><td>${info.type}</td><td>${info.isin}</td></tr>`;
        }
        html += '</tbody></table>';
    } else if (api === 'finmind') {
        if (data.chart_image) {
            html += `<img src="data:image/png;base64,${data.chart_image}" alt="股票圖表">`;
        }
        // Handle both data.data (with chart) and data (without chart) structures
        const tableData = Array.isArray(data) ? data : (data.data || []);
        html += createTable(tableData, translations);
    } else if (api === 'finmind_per') {
        html += createTable(data, translations);
    } else if (api === 'finmind_institutional') {
        html += createTable(data, translations);
    } else if (api === 'finmind_margin') {
        html += createTable(data, translations);
    } else if (api === 'finmind_revenue') {
        html += createTable(data, translations);
    } else if (api === 'finmind_weekly') {
        if (data.chart_image) {
            html += `<img src="data:image/png;base64,${data.chart_image}" alt="股票圖表">`;
        }
        html += createTable(data.data, translations);
    } else if (api === 'finmind_monthly') {
        if (data.chart_image) {
            html += `<img src="data:image/png;base64,${data.chart_image}" alt="股票圖表">`;
        }
        html += createTable(data.data, translations);
    }
    display.innerHTML = html;
}

function createTable(data, columnMap) {
    if (!Array.isArray(data)) return '<p>資料格式不支援表格顯示。</p>';
    if (data.length === 0) return '<p>無資料。</p>';
    let html = '<table border="1"><thead><tr>';
    const keys = Object.keys(data[0]);
    console.log('Table keys:', keys); // Debug log
    console.log('Column map:', columnMap); // Debug log

    // Fallbacks for zh-tw
    const fallbackMap = {
        'Trading_Volume': '成交量',
        'Trading_money': '成交金額',
        'Trading_turnover': '成交筆數',
        'close': '收盤價',
        'date': '日期',
        'max': '最高價',
        'min': '最低價',
        'open': '開盤價',
        'spread': '漲跌',
        'stock_id': '股票代碼'
    };

    for (let key of keys) {
        let displayName = columnMap[key];
        if (!displayName || displayName === key) {
            displayName = fallbackMap[key] || key;
        }
        html += `<th>${displayName}</th>`;
    }
    html += '</tr></thead><tbody>';
    for (let row of data.slice(0, 20)) {  // 顯示前20筆
        html += '<tr>';
        for (let key of keys) {
            html += `<td>${row[key] || ''}</td>`;
        }
        html += '</tr>';
    }
    html += '</tbody></table>';
    return html;
}

fetchBtn.addEventListener('click', () => {
    const api = select.value;
    const stockId = stockIdInput.value;
    fetchData(api, stockId);
});

autoRefreshCheckbox.addEventListener('change', () => {
    if (autoRefreshCheckbox.checked) {
        autoRefreshInterval = setInterval(() => {
            const api = select.value;
            const stockId = stockIdInput.value;
            fetchData(api, stockId);
        }, 10000);  // 10 seconds
    } else {
        clearInterval(autoRefreshInterval);
    }
});

// Initial fetch
fetchData(select.value, stockIdInput.value);

// 在頁面加載時獲取用戶信息
document.addEventListener('DOMContentLoaded', function() {
    fetchUserInfo();
});

function fetchUserInfo() {
    fetch('/api/finmind/user_info')
        .then(res => res.json())
        .then(data => {
            console.log('User info response:', data); // Debug log
            if (data.error) {
                document.getElementById('usage').textContent = '無法獲取統計';
            } else {
                // Try different possible field names
                const userCount = data.user_count || data.api_count || data.count || 0;
                const limit = data.api_request_limit || data.limit || data.api_limit || 600;
                document.getElementById('usage').textContent = 
                    `已使用: ${userCount} / 上限: ${limit}`;
            }
        })
        .catch(err => {
            console.error('User info error:', err); // Debug log
            document.getElementById('usage').textContent = '載入失敗';
        });
}