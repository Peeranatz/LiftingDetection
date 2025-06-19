from flask import (
    Flask,
    render_template_string,
    request,
)

from models.action_model import Action

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    # รับค่าจากฟอร์ม filter
    person_id = request.values.get("person_id", "")
    action = request.values.get("action", "")
    object_type = request.values.get("object_type", "")
    timestamp = request.values.get("timestamp", "")

    # สร้าง dictionary สำหรับเก็บเงื่อนไข query ของ action
    action_query = {}
    if person_id:
        action_query["person_id"] = person_id
    if action:
        action_query["action"] = action
    if object_type:
        action_query["object_type"] = object_type
    if timestamp:
        from datetime import datetime, timedelta
        try:
            dt = datetime.strptime(timestamp, "%Y-%m-%d")
            action_query["start_time__lte"] = dt + timedelta(days=1)
            action_query["end_time__gte"] = dt
        except Exception:
            pass

    # ดึงข้อมูล action ตามเงื่อนไข
    actions = Action.objects(**action_query).order_by("-start_time")[:20]

    return render_template_string(
        """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Action Detection Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
        <style>
            body {
                background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%);
                min-height: 100vh;
            }
            .dashboard-container {
                max-width: 1000px;
                margin: 40px auto;
                background: #fff;
                border-radius: 18px;
                box-shadow: 0 4px 24px rgba(0,0,0,0.08);
                padding: 32px 40px;
            }
            .table thead {
                background: #0288d1;
                color: #fff;
            }
            .table-striped>tbody>tr:nth-of-type(odd) {
                background-color: #e3f2fd;
            }
            .brand-header {
                display: flex;
                align-items: center;
                gap: 16px;
                margin-bottom: 24px;
            }
            .brand-header i {
                font-size: 2.5rem;
                color: #0288d1;
            }
            .brand-header h2 {
                margin: 0;
                font-weight: 700;
                color: #01579b;
            }
            .filter-form {
                background: #e0f7fa;
                border-radius: 12px;
                padding: 18px 24px;
                margin-bottom: 24px;
            }
            .filter-form label {
                font-weight: 500;
            }
        </style>
    </head>
    <body>
        <div class="dashboard-container">
            <div class="brand-header">
                <i class="fa-solid fa-truck-droplet"></i>
                <h2>Action Detection Dashboard</h2>
            </div>
            <!-- ฟอร์มสำหรับกรองข้อมูล (Filter Query) -->
            <form class="filter-form row g-3" method="get">
                <div class="col-md-3">
                    <label for="person_id" class="form-label">Person ID</label>
                    <input type="text" class="form-control" id="person_id" name="person_id" value="{{request.values.get('person_id','')}}">
                </div>
                <div class="col-md-3">
                    <label for="action" class="form-label">Action</label>
                    <input type="text" class="form-control" id="action" name="action" value="{{request.values.get('action','')}}">
                </div>
                <div class="col-md-3">
                    <label for="object_type" class="form-label">Object Type</label>
                    <input type="text" class="form-control" id="object_type" name="object_type" value="{{request.values.get('object_type','')}}">
                </div>
                <div class="col-md-3">
                    <label for="timestamp" class="form-label">วันที่</label>
                    <input type="date" class="form-control" id="timestamp" name="timestamp" value="{{request.values.get('timestamp','')}}">
                </div>
                <div class="col-12 d-flex align-items-end">
                    <button type="submit" class="btn btn-primary w-100"><i class="fa-solid fa-filter"></i> กรองข้อมูล</button>
                </div>
            </form>
            <a href="/object_count" class="btn btn-success mb-3">ไปยังหน้า Object count</a>
            <div class="row">
                <div class="col-12">
                    <h5 class="mt-3 mb-2"><i class="fa-solid fa-person-walking"></i> Action Detection (20 รายการล่าสุด)</h5>
                    <div class="table-responsive">
                        <table class="table table-striped table-hover align-middle">
                            <thead>
                                <tr>
                                    <th>Person ID</th>
                                    <th>Action</th>
                                    <th>Object Type</th>
                                    <th>Start Time</th>
                                    <th>End Time</th>
                                </tr>
                            </thead>
                            <tbody>
                                {% for a in actions %}
                                <tr>
                                    <td>{{a.person_id}}</td>
                                    <td>{{a.action}}</td>
                                    <td>{{a.object_type or '-'}}</td>
                                    <td>{{a.start_time.strftime('%d/%m/%Y %H:%M:%S')}}</td>
                                    <td>{{a.end_time.strftime('%d/%m/%Y %H:%M:%S')}}</td>
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """,
        actions=actions,
        request=request,
    )

@app.route("/object_count", methods=["GET", "POST"])
def object_count():
    from datetime import datetime
    count = 0
    objects = []
    start_time_str = ""
    end_time_str = ""
    at_time_str = ""
    object_type = ""
    if request.method == "POST":
        at_time_str = request.form.get("at_time", "")
        start_time_str = request.form.get("start_time", "")
        end_time_str = request.form.get("end_time", "")
        object_type = request.form.get("object_type", "")
        try:
            if at_time_str:
                at_time = datetime.strptime(at_time_str, "%Y-%m-%dT%H:%M")
                query = {
                    "start_time__lte": at_time,
                    "end_time__gte": at_time
                }
                if object_type:
                    query["object_type"] = object_type
                objects = Action.objects(**query)
                count = objects.count()
            elif start_time_str and end_time_str:
                start_time = datetime.strptime(start_time_str, "%Y-%m-%dT%H:%M")
                end_time = datetime.strptime(end_time_str, "%Y-%m-%dT%H:%M")
                query = {
                    "start_time__lte": end_time,
                    "end_time__gte": start_time
                }
                if object_type:
                    query["object_type"] = object_type
                objects = Action.objects(**query)
                count = objects.count()
        except Exception:
            pass
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Object Count at Time</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
        <style>
            body {
                background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%);
                min-height: 100vh;
            }
            .dashboard-container {
                max-width: 1000px;
                margin: 40px auto;
                background: #fff;
                border-radius: 18px;
                box-shadow: 0 4px 24px rgba(0,0,0,0.08);
                padding: 32px 40px;
            }
            .brand-header {
                display: flex;
                align-items: center;
                gap: 16px;
                margin-bottom: 24px;
            }
            .brand-header i {
                font-size: 2.5rem;
                color: #0288d1;
            }
            .brand-header h2 {
                margin: 0;
                font-weight: 700;
                color: #01579b;
            }
            .filter-form {
                background: #e0f7fa;
                border-radius: 12px;
                padding: 18px 24px;
                margin-bottom: 24px;
            }
            .filter-form label {
                font-weight: 500;
            }
        </style>
    </head>
    <body>
        <div class="dashboard-container">
            <div class="brand-header">
                <a href="/" class="btn btn-secondary"><i class="fa-solid fa-arrow-left"></i></a>
                <h2>ดูจำนวน Object</h2>
            </div>
            <form method="post" class="filter-form row g-2 align-items-end mb-3">
                <div class="col-auto">
                    <label for="at_time" class="form-label">เลือกวันเวลาเดียว</label>
                    <input type="datetime-local" class="form-control" id="at_time" name="at_time">
                </div>
                <div class="col-auto">
                    <label for="start_time" class="form-label">ตั้งแต่วันเวลา</label>
                    <input type="datetime-local" class="form-control" id="start_time" name="start_time">
                </div>
                <div class="col-auto">
                    <label for="end_time" class="form-label">ถึงวันเวลา</label>
                    <input type="datetime-local" class="form-control" id="end_time" name="end_time">
                </div>
                <div class="col-auto">
                    <input type="text" class="form-control" name="object_type" placeholder="Object Type (ถ้ามี)">
                </div>
                <div class="col-auto">
                    <button type="submit" class="btn btn-success">ดูจำนวน</button>
                </div>
            </form>
            {% if at_time_str %}
                <h4>
                    จำนวน Object ณ เวลา
                    <span style="color:#0288d1">{{at_time_str}}</span>
                    : <span style="color:blue">{{count}}</span>
                </h4>
            {% elif start_time_str and end_time_str %}
                <h4>
                    จำนวน Object ตั้งแต่
                    <span style="color:#0288d1">{{start_time_str}}</span>
                    ถึง
                    <span style="color:#0288d1">{{end_time_str}}</span>
                    : <span style="color:blue">{{count}}</span>
                </h4>
            {% endif %}
            {% if objects %}
                <div class="table-responsive">
                    <table class="table table-striped table-hover align-middle">
                        <thead>
                            <tr>
                                <th>Person ID</th>
                                <th>Object Type</th>
                                <th>Start Time</th>
                                <th>End Time</th>
                            </tr>
                        </thead>
                        <tbody>
                        {% for o in objects %}
                            <tr>
                                <td>{{o.person_id}}</td>
                                <td>{{o.object_type}}</td>
                                <td>{{o.start_time.strftime('%d/%m/%Y %H:%M:%S')}}</td>
                                <td>{{o.end_time.strftime('%d/%m/%Y %H:%M:%S')}}</td>
                            </tr>
                        {% endfor %}
                        </tbody>
                    </table>
                </div>
            {% endif %}
        </div>
    </body>
    </html>
    """, at_time_str=at_time_str, start_time_str=start_time_str, end_time_str=end_time_str, count=count, objects=objects)

if __name__ == "__main__":
    app.run(debug=True)