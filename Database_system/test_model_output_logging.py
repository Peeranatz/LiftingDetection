from log_to_mongo import log_action
from datetime import datetime, timedelta
from random import choice, randint
from models.action_model import Action

def random_data():
    names = ["sky", "bas", "chokun"]
    actions = ["เดิน", "ยก", "วาง", "carrying"]
    objects = ["ลัง", "กล่อง", "ขวดน้ำ", "แพ็คเครื่องดื่ม"]
    base_time = datetime(2025, 5, 26, 14, 0)

    Action.objects.delete()

    for _ in range(10):
        person = choice(names)
        action = choice(actions)
        start_offset = randint(0, 60)
        duration = randint(1, 10)
        start_time = base_time + timedelta(minutes=start_offset)
        end_time = start_time + timedelta(minutes=duration)
        object_type = choice(objects) if action == "carrying" else None
        log_action(person, action, start_time, end_time, object_type=object_type)

def test_log_and_query():
    random_data()
    actions = Action.objects()
    for a in actions:
        print("Action:", a.person_id, a.action, a.object_type, a.start_time, a.end_time)

if __name__ == "__main__":
    test_log_and_query()