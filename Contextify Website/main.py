from website import create_app
from predictor import load_model, predict

app = create_app()

if __name__ =='__main__':
    app.run(debug=True)
