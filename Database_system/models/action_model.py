from mongoengine import Document, StringField, DateTimeField, connect

connect("activity_db", host="localhost", port=27017)

class Action(Document):
    person_id = StringField(required=True)  # เพิ่ม person_id
    action = StringField(required=True)
    object_type = StringField(required=False)  # เพิ่ม object_type (optional)
    start_time = DateTimeField(required=True)
    end_time = DateTimeField(required=True)
    created_at = DateTimeField(required=True)

    meta = {"collection": "actions"}