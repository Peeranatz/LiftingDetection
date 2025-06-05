from datetime import datetime
from models.action_model import Action
from models.object_model import ObjectDetection


def log_action(person_id, action, start_time, end_time):
    # บันทึกข้อมูล action
    act = Action(
        person_id=person_id,
        action=action,
        start_time=start_time,
        end_time=end_time,
        created_at=datetime.utcnow(),
    )
    act.save()
    print("✅ Logged Action:", person_id, action)


def log_object(person_id, object_type, start_time, end_time):
    # บันทึกข้อมูล object detection
    obj = ObjectDetection(
        person_id=person_id,
        object_type=object_type,
        start_time=start_time,
        end_time=end_time,
        created_at=datetime.utcnow(),
    )
    obj.save()
    print("✅ Logged Object:", person_id, object_type)