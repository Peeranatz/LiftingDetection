from datetime import datetime
from models.action_model import Action

def log_action(person_id, action, start_time, end_time, object_type=None):
    act = Action(
        person_id=person_id,
        action=action,
        object_type=object_type,
        start_time=start_time,
        end_time=end_time,
        created_at=datetime.utcnow(),
    )
    act.save()
    print("✅ Logged Action:", person_id, action, object_type)