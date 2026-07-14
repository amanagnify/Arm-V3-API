from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from api.routes import gesture, robot, status, calibration
from api.services.websocket_service import ws_manager
from contextlib import asynccontextmanager
import asyncio
import logging

# Configure logging so the user can see connection acknowledgements in the terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:     %(message)s"
)
logger = logging.getLogger("api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API Server Starting: Attempting to connect to Robot...")
    # Startup
    await ws_manager.connect()
    yield
    # Shutdown
    logger.info("API Server Shutting Down: Closing robot connection...")
    await ws_manager.disconnect()

# Disable default docs to replace with dark mode
app = FastAPI(title="Robotic Arm Control API", lifespan=lifespan, docs_url=None)

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
    <link type="text/css" rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui.css">
    <title>Robotic Arm Control API - Swagger UI</title>
    <style>
        /* Comprehensive Dark Mode Overrides */
        body, .swagger-ui { background-color: #121212 !important; color: #e0e0e0 !important; font-family: sans-serif; }
        .swagger-ui .info .title, .swagger-ui .info p, .swagger-ui .info hgroup.main a, .swagger-ui .info .base-url { color: #e0e0e0 !important; }
        .swagger-ui .opblock .opblock-summary-operation-id, .swagger-ui .opblock .opblock-summary-path, .swagger-ui .opblock .opblock-summary-path__deprecated, .swagger-ui .opblock .opblock-summary-description { color: #e0e0e0 !important; }
        .swagger-ui .opblock-description-wrapper p, .swagger-ui .opblock-external-docs-wrapper p, .swagger-ui .opblock-title_normal p { color: #e0e0e0 !important; }
        .swagger-ui .response-col_status, .swagger-ui .response-col_description { color: #e0e0e0 !important; }
        .swagger-ui .parameter__name, .swagger-ui .parameter__type, .swagger-ui .parameter__in { color: #e0e0e0 !important; }
        .swagger-ui table thead tr td, .swagger-ui table thead tr th { color: #e0e0e0 !important; border-bottom-color: #333 !important; }
        .swagger-ui .model-title, .swagger-ui .model, .swagger-ui section.models h4, .swagger-ui .model-title__text { color: #e0e0e0 !important; }
        .swagger-ui .prop-name, .swagger-ui .prop-type, .swagger-ui .prop-format { color: #a9b7c6 !important; }
        .swagger-ui .model-box, .swagger-ui section.models .model-container { background: #1e1e1e !important; }
        .swagger-ui .opblock-body pre.microlight { background: #1e1e1e !important; color: #e0e0e0 !important; }
        .swagger-ui .scheme-container, .swagger-ui .opblock .opblock-section-header { background: #1e1e1e !important; box-shadow: none !important; border-bottom: 1px solid #333 !important; }
        
        /* Fix the white background on buttons, schemas, and interactive elements */
        .swagger-ui button, .swagger-ui .model-toggle, .swagger-ui .model-box-control, .swagger-ui .expand-operation { background: transparent !important; color: #e0e0e0 !important; border: none !important; box-shadow: none !important; }
        .swagger-ui .btn { background: transparent !important; border: 1px solid #666 !important; color: #e0e0e0 !important; }
        .swagger-ui .btn-clear { color: #ff6b6b !important; }
        
        .swagger-ui select, .swagger-ui input, .swagger-ui textarea { background: #121212 !important; color: #e0e0e0 !important; border: 1px solid #444 !important; }
        .swagger-ui .dialog-ux .modal-ux { background: #1e1e1e !important; color: #e0e0e0 !important; border: 1px solid #444 !important;}
        .swagger-ui .dialog-ux .modal-ux-header h3 { color: #e0e0e0 !important; }
        .swagger-ui .opblock-tag { color: #e0e0e0 !important; border-bottom-color: #444 !important;}
        .swagger-ui .opblock.opblock-post { background: rgba(73,204,144,.1) !important; border-color: rgba(73,204,144,.3) !important; }
        .swagger-ui .opblock.opblock-get { background: rgba(97,175,254,.1) !important; border-color: rgba(97,175,254,.3) !important; }
        svg { fill: #e0e0e0 !important; }
    </style>
    </head>
    <body>
    <div id="swagger-ui"></div>
    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui-bundle.js"></script>
    <script>
    const ui = SwaggerUIBundle({
        url: '/openapi.json',
        dom_id: '#swagger-ui',
        presets: [ SwaggerUIBundle.presets.apis, SwaggerUIBundle.SwaggerUIStandalonePreset ],
        layout: "BaseLayout"
    })
    </script>
    </body>
    </html>
    """)

app.include_router(gesture.router)
app.include_router(robot.router)
app.include_router(status.router)
app.include_router(calibration.router)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
