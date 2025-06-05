from log_to_mongo import log_action, log_object
from datetime import datetime, timedelta
from random import choice, randint
from models.action_model import Action
from models.object_model import ObjectDetection


def random_data():
    names = ["sky", "bas", "chokun"]
    actions = ["ยก", "วาง", "เดิน", "หยิบ"]
    objects = ["ลัง", "กล่อง", "ขวดน้ำ", "แพ็คเครื่องดื่ม"]
    base_time = datetime(2025, 5, 26, 14, 0)

    # ลบข้อมูลเก่าเพื่อความสะอาด
    Action.objects.delete()
    ObjectDetection.objects.delete()

    # สุ่มสร้างข้อมูล 10 รายการ
    for _ in range(10):
        person = choice(names)
        action = choice(actions)
        object_type = choice(objects)
        # สุ่มเวลาเริ่มและจบ (แต่ละรายการห่างกัน 0-60 นาที)
        start_offset = randint(0, 60)
        duration = randint(1, 10)
        start_time = base_time + timedelta(minutes=start_offset)
        end_time = start_time + timedelta(minutes=duration)
        # log action และ object
        log_action(person, action, start_time, end_time)
        log_object(person, object_type, start_time, end_time)


def test_log_and_query():
    random_data()  # สร้างข้อมูลสุ่ม

    # Query ข้อมูล action
    actions = Action.objects()
    for a in actions:
        print("Action:", a.person_id, a.action, a.start_time, a.end_time)

    # Query ข้อมูล object
    objects = ObjectDetection.objects()
    for o in objects:
        print("Object:", o.person_id, o.object_type, o.start_time, o.end_time)


if __name__ == "__main__":
    test_log_and_query()