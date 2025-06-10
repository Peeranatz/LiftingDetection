from datetime import datetime
from models.action_model import Action
from models.object_model import ObjectDetection

def log_action(action, start_time, end_time):
    # บันทึกข้อมูล action
    act = Action(
        action=action,
        start_time=start_time,
        end_time=end_time,
        created_at=datetime.utcnow(),
    )
    act.save()
    print("✅ Logged Action:", action)

def log_object(person_id, object_type, start_time, end_time, box_id=None):
    # บันทึกข้อมูล object detection
    obj = ObjectDetection(
        person_id=person_id,
        object_type=object_type,
        box_id=box_id,
        start_time=start_time,
        end_time=end_time,
        created_at=datetime.utcnow(),
    )
    obj.save()
    print("✅ Logged Object:", person_id, object_type, box_id)