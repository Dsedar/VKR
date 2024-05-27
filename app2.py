import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State
import plotly.express as px
from PIL import Image
import numpy as np
import base64
import io
import torch
from torchvision.transforms import functional as F
from utils.inference import predict_image, load_model, display_predictions, create_xml_annotation, parse_filename, calculate_geo_coordinates
import re

wordname_50 = ['Other Ship', 'Other Warship', 'Submarine', 'Other Aircraft Carrier', 'Enterprise', 
                            'Nimitz', 'Midway', 'Ticonderoga', 'Other Destroyer', 'Atago DD', 'Arleigh Burke DD', 
                            'Hatsuyuki DD', 'Hyuga DD', 'Asagiri DD', 'Other Frigate', 'Perry FF', 'Patrol', 
                            'Other Landing', 'YuTing LL', 'YuDeng LL', 'YuDao LL', 'YuZhao LL', 'Austin LL', 
                            'Osumi LL', 'Wasp LL', 'LSD 41 LL', 'LHA LL', 'Commander', 'Other Auxiliary Ship', 
                            'Medical Ship', 'Test Ship', 'Training Ship', 'AOE', 'Masyuu AS', 'Sanantonio AS', 'EPF', 
                            'Other Merchant', 'Container Ship', 'RoRo', 'Cargo', 'Barge', 'Tugboat', 'Ferry', 'Yacht', 
                            'Sailboat', 'Fishing Vessel', 'Oil Tanker', 'Hovercraft', 'Motorboat', 'Dock']

wordname_50_rus = ['Другой корабль', 'Другой военный корабль', 'Подводная лодка', 'Другой авианосец', 'Enterprise', 
                            'Nimitz', 'Midway', 'Ticonderoga', 'Другой эсминец', 'Atago DD', 'Arleigh Burke DD', 
                            'Hatsuyuki DD', 'Hyuga DD', 'Asagiri DD', 'Другой фрегат', 'Perry FF', 'Патрульное судно', 
                            'Другой десантный', 'YuTing LL', 'YuDeng LL', 'YuDao LL', 'YuZhao LL', 'Austin LL', 
                            'Osumi LL', 'Wasp LL', 'LSD 41 LL', 'LHA LL', 'Командирское судно', 'Другой вспомогательный', 
                            'Медицинское судно', 'Испытательное судно', 'Учебное судно', 'AOE', 'Masyuu AS', 'Sanantonio AS', 'EPF', 
                            'Другой торговый', 'Контейнеровоз', 'RoRo', 'Грузовой', 'Баржа', 'Буксирное судно', 'Паром', 'Яхта', 
                            'Парусная лодка', 'Рыболовное судно', 'Нефтяной танкер', 'Судно на воздушной подушке', 'Моторная лодка', 'Док']

warships = ['Other Warship', 'Other Destroyer', 'Other Frigate', 'Other Aircraft Carrier', 'Enterprise', 'Submarine', 'Nimitz', 'Other Landing',
            'Midway', 'Ticonderoga', 'Hatsuyuki DD', 'Hyuga DD', 'Asagiri DD', 'Atago DD', 'Arleigh Burke DD', 'Perry FF', 'Patrol',
            'YuTing LL', 'YuDeng LL', 'YuDao LL', 'YuZhao LL', 'LSD 41 LL', 'LHA LL', 'Masyuu AS', 'Sanantonio AS', 'EPF', 'Austin LL',
            'Osumi LL', 'Wasp LL', 'Commander', 'Other Auxiliary Ship', 'AOE', 'Test Ship', 'Training Ship', 'Medical Ship']

merchant_ships = ['Other Merchant', 'Container Ship', 'RoRo', 'Cargo', 'Barge', 'Tugboat', 'Ferry', 'Yacht', 
                  'Sailboat', 'Fishing Vessel', 'Oil Tanker', 'Hovercraft', 'Motorboat', 'Dock']

other_ships = ['Other Ship']

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'Ship Detection'

# Load the model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = load_model('checkpoint/ship_detect_rsnn_10epochs_8batch.h5', device)
model.eval()

# Define the layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Ship Detection App", className='text-center text-primary mb-4'), width=12)
    ]),
    dbc.Row([
        dbc.Col([
            html.H4("Загрузка изображения"),
            dcc.Upload(
                id='upload-image',
                children=html.Div([
                    'Перетащите изображение сюда или ',
                    html.A('выберите')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin-top': '25px'
                },
                multiple=False
            ),
            html.Div(id='output-image-upload', style={'margin': '15px'}),
            html.Button('Создать XML аннотацию', id='create-xml-button', n_clicks=0, className='btn btn-primary mt-3', style={'margin': '15px'})
        ], width=6),
        dbc.Col([
            html.H4("Фильтр по типу кораблей"),
            dbc.ButtonGroup(
                [
                    dbc.Button("Все корабли", id='all-ships-btn', n_clicks=0, className='btn btn-secondary'),
                    dbc.Button("Военные корабли", id='warships-btn', n_clicks=0, className='btn btn-secondary'),
                    dbc.Button("Торговые и гражданские объекты", id='merchant-ships-btn', n_clicks=0, className='btn btn-secondary'),
                    dbc.Button("Другие", id='other-ships-btn', n_clicks=0, className='btn btn-secondary')
                ],
                className='mt-3', style={'width': '100%', 'margin-top': '15px'}
            ),
            html.Div(id='detection-results', style={'margin-top': '15px', 'whiteSpace': 'pre-wrap'}),
        ], width=6),
    ]),
    dbc.Row([
        dbc.Col(html.Div(id='message', className='mt-3 text-center text-info'), width=12)
    ]),
])

def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return Image.open(io.BytesIO(decoded))

def parse_filename(filename):
    pattern = re.compile(r'(?P<lat>-?\d+\.\d+)_(?P<lon>-?\d+\.\d+)_(?P<scale>\d+\.\d+)(_(?P<direction>\w+))?')
    match = pattern.match(filename)
    if not match:
        raise ValueError("Изображение не содержит координаты, масштаб, и направление в названии")
    
    lat = float(match.group('lat'))
    lon = float(match.group('lon'))
    scale = float(match.group('scale'))
    direction = match.group('direction')
    
    return lat, lon, scale, direction

@app.callback(
    [Output('output-image-upload', 'children'),
     Output('detection-results', 'children'),
     Output('message', 'children')],
    [Input('upload-image', 'contents'),
     Input('create-xml-button', 'n_clicks'),
     Input('all-ships-btn', 'n_clicks'),
     Input('warships-btn', 'n_clicks'),
     Input('merchant-ships-btn', 'n_clicks'),
     Input('other-ships-btn', 'n_clicks')],
    [State('upload-image', 'filename'),
     State('upload-image', 'last_modified')]
)
def update_output(contents, n_clicks_xml, n_clicks_all, n_clicks_war, n_clicks_merch, n_clicks_other, filename, date):
    if contents is not None:
        image = parse_contents(contents)
        image = image.convert("RGB")  # Convert to RGB mode
        image_path = 'uploaded_image.jpg'
        image.save(image_path)

        message = ""
        # Check and parse filename for coordinates and scale
        try:
            lat, lon, scale, direction = parse_filename(filename)
            coords_available = True
            message = "Название содержит координаты, масштаб и направление."
        except ValueError:
            lat, lon, scale, direction = None, None, None, None
            coords_available = False
            message = "Название не содержит координаты, масштаб и направление. Применяется обнаружение без определения координат."

        # Detect ships
        pred_boxes, pred_labels, pred_scores = predict_image(model, image_path, device, confidence_threshold=0.4)

        # Display the uploaded image with bounding boxes
        filtered_boxes, filtered_labels, filtered_scores = pred_boxes, pred_labels, pred_scores
        selected_group = 'all'
        ctx = dash.callback_context
        if ctx.triggered:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if button_id == 'warships-btn':
                selected_group = 'warships'
                filtered_boxes, filtered_labels, filtered_scores = zip(*[(box, label, score) for box, label, score in zip(pred_boxes, pred_labels, pred_scores) if wordname_50[label - 1] in warships])
            elif button_id == 'merchant-ships-btn':
                selected_group = 'merchant_ships'
                filtered_boxes, filtered_labels, filtered_scores = zip(*[(box, label, score) for box, label, score in zip(pred_boxes, pred_labels, pred_scores) if wordname_50[label - 1] in merchant_ships])
            elif button_id == 'other-ships-btn':
                selected_group = 'other_ships'
                filtered_boxes, filtered_labels, filtered_scores = zip(*[(box, label, score) for box, label, score in zip(pred_boxes, pred_labels, pred_scores) if wordname_50[label - 1] in other_ships])
            elif 'ship-class-btn' in button_id:
                selected_class = button_id.split('.')[0].split('index":"')[1].split('"}')[0]
                filtered_boxes, filtered_labels, filtered_scores = zip(*[(box, label, score) for box, label, score in zip(pred_boxes, pred_labels, pred_scores) if wordname_50[label - 1] == selected_class])

        fig = display_predictions(image_path, filtered_boxes, filtered_labels, filtered_scores, wordname_50)
        graph = dcc.Graph(figure=fig)

        # Create XML annotation if button clicked
        if n_clicks_xml > 0:
            xml_path = 'annotation.xml'
            create_xml_annotation(xml_path, image_path, pred_boxes, pred_labels, pred_scores, wordname_50, lat, lon, scale, direction)

        # Print detection results with geo coordinates if available
        detection_info = []
        img = Image.open(image_path)
        img_size = img.size
        for box, label, score in zip(filtered_boxes, filtered_labels, filtered_scores):
            if coords_available:
                geo_lat, geo_lon = calculate_geo_coordinates(box, img_size, lat, lon, scale, direction)
                detection_info.append({
                    "class": wordname_50_rus[label - 1],
                    "score": score,
                    "geo_lat": geo_lat,
                    "geo_lon": geo_lon
                })
            else:
                detection_info.append({
                    "class": wordname_50_rus[label - 1],
                    "score": score
                })

        detection_results_table = html.Table([
            html.Thead(html.Tr([html.Th("Класс"), html.Th("Широта"), html.Th("Долгота")])),
            html.Tbody([
                html.Tr([
                    html.Td(info['class']),
                    html.Td(f"{info['geo_lat']:.6f}" if coords_available else "N/A"),
                    html.Td(f"{info['geo_lon']:.6f}" if coords_available else "N/A")
                ]) for info in detection_info
            ])
        ], className='table table-striped')

        return graph, detection_results_table, message

    return None, None, 'Пожалуйста загрузите изображение'


if __name__ == '__main__':
    app.run_server(debug=True)


