import pandera as pa

EVAL_CSV_SCHEMA = pa.DataFrameSchema(
    {
        "text": pa.Column(pa.String),
        "label": pa.Column(pa.String),
    }
)

INFERENCE_CSV_SCHEMA = pa.DataFrameSchema(
    {
        "text": pa.Column(pa.String),
    }
)

LABEL_DEF = {
    "translate": 0,
    "travel_alert": 1,
    "flight_status":2,
    "lost_luggage": 3,
    "travel_suggestion": 4,
    "carry_on": 5,
    "book_hotel":6,
    "book_flight": 7,
    "out_of_scope": 8}