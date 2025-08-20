from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from .utils import CFG_PATH, set_yaml_entry, get_yaml_entry

router = APIRouter()


STATUS_OK, STATUS_FAIL = 'success', 'failure'
MSG_OK, MSG_FAIL = 'Data processed', 'Data could not be processed'


def generate_html_template(title, buttons):
    """
    Generate a minimal HTML template with buttons that link to FastAPI routes.
    
    Args:
        title (str): The title for the webpage
        buttons (list): A list of dictionaries with 'name' and 'route' keys
                        e.g. [{'name': 'Submit', 'route': '/submit'}, ...]
    
    Returns:
        str: HTML template as a string
    """
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        .button-container {{
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 10px;
            margin-top: 20px;
        }}
        button {{
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s;
        }}
        button:hover {{
            background-color: #45a049;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="button-container">
"""
    
    # Add buttons with JavaScript to call the FastAPI routes
    for button in buttons:
        html += f"""
        <button onclick="callRoute('{button['route']}')">{button['name']}</button>"""
    
    # Add JavaScript to handle button clicks
    html += """
    </div>

    <script>
        async function callRoute(route) {
            try {
                const response = await fetch(route, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
                
                const result = await response.json();
                console.log('Success:', result);
                alert('Action completed successfully!');
            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred: ' + error);
            }
        }
    </script>
</body>
</html>
"""
    
    return html


@router.get("/", response_class=HTMLResponse)
async def get_html_page():
    buttons = [
        {"name": "Распознавание по запросу", "route": "/turn-off-loop"},
        {"name": "Распознавание каждые 5 сек", "route": "/loop-5-sec"},
        {"name": "Распознавание каждые 3 сек", "route": "/loop-3-sec"},
        {"name": "Распознавание каждую 1 сек", "route": "/loop-1-sec"},
        {"name": "Распознавание непрерывное", "route": "/loop-continuous"},
        {"name": "Модель TensorRT FP16", "route": "/use-tensorrt-16"},
        {"name": "Модель TensorRT FP32", "route": "/use-tensorrt-32"},
        {"name": "Модель без оптимизации", "route": "/use-original"},
        {"name": "Запись на диск", "route": "/with-dumps"},
        {"name": "Без записи на диск", "route": "/no-dumps"},
        {"name": "Использовать пример фотки", "route": "/use-dummy"},
        {"name": "Использовать реальные фотки", "route": "/use-real"}
    ]
    
    html_content = generate_html_template("Конфигурация", buttons)
    return html_content


def try_fail_action(meth, args):
    resp = {"status": STATUS_OK, "message": MSG_OK}
    try:
        meth(*args)
    except:
        resp = {"status": STATUS_FAIL, "message": MSG_FAIL}
    finally:
        return resp


@router.post("/turn-off-loop")
async def turn_off_loop():
    try_fail_action(set_yaml_entry, [CFG_PATH, ['LOOP_SECONDS'], -1])


@router.post("/loop-5-sec")
async def turn_off_loop():
    try_fail_action(set_yaml_entry, [CFG_PATH, ['LOOP_SECONDS'], 5])


@router.post("/loop-3-sec")
async def loop_3_sec():
    try_fail_action(set_yaml_entry, [CFG_PATH, ['LOOP_SECONDS'], 3])


@router.post("/loop-1-sec")
async def loop_1_sec():
    try_fail_action(set_yaml_entry, [CFG_PATH, ['LOOP_SECONDS'], 1])


@router.post("/loop-continuous")
async def loop_continuous():
    try_fail_action(set_yaml_entry, [CFG_PATH, ['LOOP_SECONDS'], 0])


@router.post("/use-tensorrt-16")
async def use_tensorrt_16():
    try_fail_action(set_yaml_entry, [CFG_PATH, ['MODEL_NAME'], 'segment-cucumber-trt-fp16.engine'])


@router.post("/use-tensorrt-32")
async def use_tensorrt_32():
    try_fail_action(set_yaml_entry, [CFG_PATH, ['MODEL_NAME'], 'segment-cucumber-trt-fp32.engine'])


@router.post("/use-original")
async def use_original():
    try_fail_action(set_yaml_entry, [CFG_PATH, ['MODEL_NAME'], 'segment-cucumber-orig.pt'])


@router.post("/with-dumps")
async def with_dumps():
    try_fail_action(set_yaml_entry, [CFG_PATH, ['WITH_DUMPS'], True])


@router.post("/no-dumps")
async def no_dumps():
    try_fail_action(set_yaml_entry, [CFG_PATH, ['WITH_DUMPS'], False])


@router.post("/use-dummy")
async def use_dummy():
    try_fail_action(set_yaml_entry, [CFG_PATH, ['USE_DUMMY'], True])


@router.post("/use-real")
async def use_real():
    try_fail_action(set_yaml_entry, [CFG_PATH, ['USE_DUMMY'], False])


# Example usage with FastAPI:
"""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def get_html_page():
    buttons = [
        {"name": "Process Data", "route": "/process"},
        {"name": "Generate Report", "route": "/generate-report"},
        {"name": "Clear Cache", "route": "/clear-cache"}
    ]
    
    html_content = generate_html_template("Dashboard Controls", buttons)
    return html_content

@app.post("/process")
async def process_data():
    # Process data logic here
    return {"status": "success", "message": "Data processed"}

@app.post("/generate-report")
async def generate_report():
    # Report generation logic here
    return {"status": "success", "message": "Report generated"}

@app.post("/clear-cache")
async def clear_cache():
    # Cache clearing logic here
    return {"status": "success", "message": "Cache cleared"}
"""
