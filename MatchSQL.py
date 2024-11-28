import psycopg2

# 連接到 PostgreSQL 資料庫
conn = psycopg2.connect(
    dbname="cmsdb",  # 請填入你的資料庫名稱
    user="cmsuser",             # 請填入你的 PostgreSQL 用戶名
    password="cmsuser",     # 請填入你的密碼
    host="localhost",             # 如果是本地連接則使用 localhost，若是遠程資料庫則替換為對應的 IP 或主機名稱
    port="5432"                   # PostgreSQL 預設端口為 5432，若是其他端口則修改
)

# 創建游標對象
cur = conn.cursor()

# 查詢 submission_results 表中的 submission_id 和 score
cur.execute("SELECT submission_id, score FROM submission_results")

# 讀取查詢結果
submission_result = cur.fetchall()

# 根據 submission_id 查詢 submissions 表中的 timestamp
for result in submission_result:
    submission_id = result[0]
    
    # 查詢 submissions 表中的 timestamp
    cur.execute("SELECT timestamp FROM submissions WHERE id = %s", (submission_id,))
    timestamp_result = cur.fetchone()
    
    if timestamp_result:
        timestamp = timestamp_result[0]
        print(f"submission_id: {submission_id}, score: {result[1]}, timestamp: {timestamp}")
    else:
        print(f"submission_id: {submission_id} 找不到對應的 timestamp")

# 關閉游標和連接
cur.close()
conn.close()
