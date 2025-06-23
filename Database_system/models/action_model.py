from mongoengine import Document, StringField, DateTimeField, connect

connect("activity_db", host="localhost", port=27017)

class Action(Document):
    person_id = StringField(required=True)
    action = StringField(required=True)
    object_type = StringField(required=False)
    object_id = StringField(required=False)  # เพิ่ม object_id (optional)
    start_time = DateTimeField(required=True)
    end_time = DateTimeField(required=True)
    created_at = DateTimeField(required=True)

    meta = {"collection": "actions"}