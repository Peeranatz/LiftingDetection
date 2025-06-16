from datetime import datetime
from log_to_mongo import log_action

# -------------------------------
# ส่วนที่ 1: รันโมเดลและเตรียม input
# TODO: ใส่โค้ดรันโมเดลของคุณที่นี่
# ตัวอย่าง output ที่ควรได้จากโมเดล (ควรเป็น list ของ dict)
# output_results = [
#     {
#         "person_id": "sky",
#         "action": "carrying",
#         "start_time": datetime(2025, 6, 10, 10, 0),
#         "end_time": datetime(2025, 6, 10, 10, 5),
#         "object_type": "ลัง"  # มี object_type เฉพาะ action ที่เป็น carrying
#     },
#     ...
# ]
output_results = []  # ← ลบ [] แล้วแทนที่ด้วยผลลัพธ์จริงจากโมเดล

# -------------------------------
# ส่วนที่ 2: บันทึกผลลัพธ์ลง Database

for result in output_results:
    log_action(
        result["person_id"],
        result["action"],
        result["start_time"],
        result["end_time"],
        object_type=result.get("object_type")  # ถ้าไม่มี object_type จะเป็น None
    )

print("✅ บันทึกข้อมูลจากโมเดลเสร็จสิ้น")