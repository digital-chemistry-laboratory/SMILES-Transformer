import pandas as pd
import dash
import plotly.graph_objs as go
from CGRtools.files import SMILESRead
from dash import Dash, dcc, html, Input, Output, no_update, callback, State, ctx
import base64
from io import StringIO
import pyperclip
import plotly.io as pio

# Create a Dash app instance
app = dash.Dash(__name__)
cgr_parser = SMILESRead.create_parser(ignore=True)


# df['hover_text'] = df['cgr'].apply(lambda svg: f'<div>{svg}</div>')
def parse_cgr(smiles_cgr: str):
    m = cgr_parser(smiles_cgr)
    # m = m.compose()
    m.clean2d()
    return m.depict()


def parse_file(df):
    df["cgr_img"] = (
        df["smiles_CGR"].apply(lambda x: parse_cgr(x)).drop(columns=["Unnamed: 0"])
    )
    df["smiles_img"] = df["original_smiles"].apply(lambda x: parse_cgr(x))
    df["color"] = 1
    return df


app.layout = html.Div(
    children=[
        dcc.Store(id="memory"),
        html.Div(
            className="row",
            children=[
                html.P("Color out points containing:"),
                html.Div(className="spacer"),
                dcc.Input(
                    id="string-input",
                    type="text",
                    debounce=True,
                    placeholder="[H]",
                    autoComplete="off",
                ),
            ],
            style={"display": "flex", "alignItems": "center", "gap": "10px"},
        ),
        dcc.Upload(
            id="upload-data",
            children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            # Allow multiple files to be uploaded
            multiple=False,
        ),
        dcc.Graph(id="graph"),
        dcc.Tooltip(id="graph-tooltip", background_color="rgba(255,255,255, 0.8)"),
        html.Button("Save Graph", id="save-graph-btn", n_clicks=0),
        dcc.Download(id="download-graph"),
    ]
)


def prepare_graph(df):
    scatters = []
    for color in df["color"].unique():
        df_select = df[df["color"] == color]
        scatter = go.Scatter(
            x=df_select["true"],
            y=df_select["pred"],
            mode="markers",
            name=color,
        )
        scatters.append(scatter)

    xy_line = go.Scatter(
        x=[df["true"].min(), df["true"].max()],
        y=[df["pred"].min(), df["pred"].max()],
        mode="lines",
        line=dict(color="black", dash="dash"),
    )
    fig = go.Figure(data=[*scatters, xy_line])

    fig.update_layout(
        xaxis=dict(title="True Values"),
        yaxis=dict(title="Predicted Values"),
        plot_bgcolor="rgba(255,255,255,0.9)",
        autosize=True,
    )
    fig.update_yaxes(automargin=True)
    fig.update_traces(hoverinfo="none", hovertemplate=None)
    return fig


@app.callback(
    Output("graph", "figure"),
    Output("memory", "data"),
    Input("upload-data", "contents"),
    Input("string-input", "value"),
    State("memory", "data"),
)
def update_graph(file, input_str, data):
    cgr_col = "smiles_CGR"
    trigger = ctx.triggered_id
    if file is None:
        raise dash.exceptions.PreventUpdate
    if trigger == "upload-data":
        decoded = base64.b64decode(file.split(",")[-1])
        df = pd.read_csv(StringIO(decoded.decode("utf-8")))
        df = parse_file(df)
        df["color"] = "data points"
    elif trigger == "string-input":
        df = pd.DataFrame.from_dict(data)
    if input_str is not None and len(df):
        df["color"] = (
            df[cgr_col]
            .apply(lambda x: input_str in x)
            .map({True: f"'{input_str}' present", False: f"'{input_str}' absent"})
        )

    fig = prepare_graph(df)
    return fig, df.to_dict("records")


@callback(
    Output("graph-tooltip", "show"),
    Output("graph-tooltip", "bbox"),
    Output("graph-tooltip", "children"),
    Input("graph", "hoverData"),
    State("memory", "data"),
)
def display_hover(hoverData, data):
    if hoverData is None:
        return False, no_update, no_update

    pt = hoverData["points"][0]
    bbox = pt["bbox"]
    point_index = pt["pointNumber"]
    data_point = data[point_index]

    cgr_svg_img = data_point["cgr_img"]
    smiles_svg_img = data_point["smiles_img"]
    cgr = data_point["smiles_CGR"]
    smiles = data_point["original_smiles"]
    y_true = data_point["true"]
    y_pred = data_point["pred"]
    pyperclip.copy(f"CGR: {cgr}%%%% SMILES: {smiles}")

    children = [
        html.Div(
            [
                dcc.Markdown(smiles_svg_img, dangerously_allow_html=True),
                html.P(smiles),
                dcc.Markdown(cgr_svg_img, dangerously_allow_html=True),
                html.P(cgr),
                html.P(f"True: {y_true}"),
                html.P(f"Predicted: {y_pred}"),
            ],
        )
    ]

    return True, bbox, children


# New callback for saving the graph
@app.callback(
    Output("download-graph", "data"),
    Input("save-graph-btn", "n_clicks"),
    State("graph", "figure"),
    prevent_initial_call=True,
)
def save_graph(n_clicks, fig):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate  # from dash.exceptions import PreventUpdate
    img_data = pio.to_image(
        go.Figure(fig), format="png", width=1920, height=1080, scale=2
    )
    return dict(content=img_data, filename="high_res_graph.png")


# Run the app
if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=8050)
