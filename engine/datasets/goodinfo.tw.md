要在台灣股市中執行「基本面粗篩」，最推薦的工具是 **Goodinfo! 台灣股市資訊網** 或 **財報狗**。這類網站免費、數據更新快，且能自訂多重條件。

以下我將以 **Goodinfo!** 為例，帶你一步步完成 V3.0 所需的基本面篩選並匯出 CSV。

---

### 第一階段：Goodinfo! 網頁篩選步驟

#### Step 1: 進入篩選頁面
開啟 [Goodinfo! 台灣股市資訊網](https://goodinfo.tw/tw/index.asp)，在左側選單點選 **「策略選股」** $\rightarrow$ **「類別選股」** 或直接進入 **「指標篩選」**。

#### Step 2: 設定篩選條件 (符合 V3.0 標準)
在篩選介面中，請依序加入以下四個條件：
1.  **獲利能力：** 選擇 `股東權益報酬率 (ROE)` 或 `總資產報酬率 (ROA)`。雖然我們目標是 ROIC，但在 Goodinfo 免費介面中，可以用 **「近一季 ROE > 4%」**（換算年化約 16%）作為替代指標。
2.  **經營效率：** 選擇 **「營業利益率 > 10%」**。這能確保公司本業賺錢，不是靠賣土地或業外收入。
3.  **財務體質：** 選擇 **「負債比率 < 45%」**。這能過濾掉財務槓桿過高的公司。
4.  **現金流：** 選擇 **「近三年每股自由現金流 > 0」**（或平均為正值）。
5. **Link**: [篩選條件](https://goodinfo.tw/tw/StockList.asp?MARKET_CAT=%E8%87%AA%E8%A8%82%E7%AF%A9%E9%81%B8&INDUSTRY_CAT=%E6%88%91%E7%9A%84%E6%A2%9D%E4%BB%B6&FL_ITEM0=%E5%96%AE%E5%AD%A3%E2%80%93ROE%28%25%29%E2%80%93%E6%9C%AC%E5%AD%A3%E5%BA%A6&FL_VAL_S0=4&FL_VAL_E0=100&FL_ITEM1=%E5%96%AE%E5%AD%A3%E2%80%93%E7%8F%BE%E9%87%91%E6%AF%94%28%25%29+%28%E7%8F%BE%E9%87%91%2F%E6%B5%81%E5%8B%95%E8%B2%A0%E5%82%B5%E6%AF%94%29%E2%80%93%E6%9C%AC%E5%AD%A3%E5%BA%A6&FL_VAL_S1=45&FL_VAL_E1=&FL_ITEM2=%E5%96%AE%E5%AD%A3%E2%80%93%E7%87%9F%E6%A5%AD%E5%88%A9%E7%9B%8A%E7%8E%87%28%25%29%E2%80%93%E6%9C%AC%E5%AD%A3%E5%BA%A6&FL_VAL_S2=10&FL_VAL_E2=100&FL_ITEM3=%E8%BF%91%E5%9B%9B%E5%AD%A3%E2%80%93%E8%87%AA%E7%94%B1%E7%8F%BE%E9%87%91%E6%B5%81%E9%87%8F%28%E5%84%84%29&FL_VAL_S3=0&FL_VAL_E3=&FL_ITEM4=&FL_VAL_S4=&FL_VAL_E4=&FL_ITEM5=&FL_VAL_S5=&FL_VAL_E5=&FL_ITEM6=&FL_VAL_S6=&FL_VAL_E6=&FL_ITEM7=&FL_VAL_S7=&FL_VAL_E7=&FL_ITEM8=&FL_VAL_S8=&FL_VAL_E8=&FL_ITEM9=&FL_VAL_S9=&FL_VAL_E9=&FL_ITEM10=&FL_VAL_S10=&FL_VAL_E10=&FL_ITEM11=&FL_VAL_S11=&FL_VAL_E11=&FL_RULE0=&FL_RULE1=&FL_RULE2=&FL_RULE3=&FL_RULE4=&FL_RULE5=&FL_RANK0=&FL_RANK1=&FL_RANK2=&FL_RANK3=&FL_RANK4=&FL_RANK5=&FL_FD0=&FL_FD1=&FL_FD2=&FL_FD3=&FL_FD4=&FL_FD5=&FL_SHEET=%E5%AD%A3%E7%B4%AF%E8%A8%88%E7%8D%B2%E5%88%A9%E8%83%BD%E5%8A%9B&FL_SHEET2=%E7%8D%B2%E5%88%A9%E8%83%BD%E5%8A%9B&FL_MARKET=%E4%B8%8A%E5%B8%82%2F%E4%B8%8A%E6%AB%83&FL_QRY=%E6%9F%A5++%E8%A9%A2)


#### Step 3: 執行篩選並檢查結果
點擊「開始篩選」，網站會列出目前符合所有條件的股票清單。

#### Step 4: 匯出資料為 CSV
1.  在結果清單上方尋找 **「匯出 Excel」** 或 **「複製到剪貼簿」** 的按鈕。
2.  如果是免費會員，通常可以直接選取網頁上的表格內容（從股票代號到最後一欄），直接複製（Ctrl+C）。

---

### 第二階段：準備 Python 所需的 CSV 檔案

現在你需要將網頁上的資料轉換成 Python 腳本能讀取的格式。

#### Step 5: 整理 Excel/CSV 格式
1.  開啟 Excel 或 Google 試算表，貼上剛才複製的內容。
2.  **最關鍵的一步：** 確保第一列（Header）有一欄名稱叫做 **`stock_id`**。
3.  將股票代號填入該欄（例如：2330, 3034, 3017...）。
4.  將檔案另存為 **`fundamental_pass.csv`**，並放在與你的 `screener_v3.py` 同一個資料夾下。



---

### 第三階段：執行 Python 腳本 (V3.0 精篩)

當你的資料夾裡已經有了 `indicators.py`、`screener_v3.py` 和 `fundamental_pass.csv` 後，按照以下步驟執行：

#### Step 6: 填寫 Token 並執行
1.  打開 `screener_v3.py`，找到這行：
    `MY_TOKEN = "你的 FinMind Token"`
2.  開啟終端機 (Terminal / Command Prompt)，進入該路徑：
    ```bash
    python screener_v3.py
    ```

#### Step 7: 判讀輸出結果
程式會開始逐一抓取這幾檔「資優生」的技術面，並計算盈虧比。
* **如果符合：** 螢幕會顯示「🎯 V3.0 攻擊型防禦標的名單」，並列出建議的**支撐價**與**目標價**。
* **如果不符合：** 程式會告訴你盈虧比不足（代表現在太貴，追高風險大）。

---

### 💡 進階小撇步：如果你不想手動輸入 stock_id
如果你從 Goodinfo 下載的 CSV 欄位名稱是「股票代碼」或「代號」，你只需要修改 `screener_v3.py` 裡面的一行程式碼即可：

```python
# 原本是：
stocks = candidate_df['stock_id'].astype(str).tolist()

# 改成你 CSV 實際的欄位名稱：
stocks = candidate_df['股票代號'].astype(str).tolist() 
```

**這樣就完成了！你現在擁有了一套「基本面網頁篩選」+「技術面 Python 精篩」的半自動化交易系統。如果你在執行過程中遇到 `ImportError` 或欄位對不上的問題，隨時把錯誤訊息貼給我！**