# Movie Recommendation System

A full-stack movie recommendation app using Flask, scikit-learn, pandas, and a lightweight HTML/CSS/JavaScript frontend.

The app recommends movies from the MovieLens dataset using a hybrid approach:

- Collaborative filtering with a trained KNN model
- Content-based similarity from movie genres
- TMDB metadata lookup for posters, ratings, release year, and overviews

## Features

- Movie autocomplete search
- Hybrid movie recommendations
- Poster grid with hover details
- TMDB metadata enrichment
- Debounced frontend search requests
- Cancellable stale frontend requests
- Path-safe backend startup from different working directories
- Faster backend startup by avoiding unnecessary large dataset/model loads at runtime

## Tech Stack

- Python
- Flask
- Flask-CORS
- pandas
- NumPy
- SciPy
- scikit-learn
- joblib
- requests
- python-dotenv
- HTML
- CSS
- JavaScript
- TMDB API
- MovieLens dataset

## Project Structure

```text
movie-recommendation-system/
├── app/
│   ├── app.py          # CLI recommender entry point
│   └── server.py       # Flask API used by the frontend
├── data/
│   ├── movie.csv       # Movie metadata
│   ├── rating.csv      # User ratings used for training
│   ├── link.csv        # MovieLens to TMDB/IMDB mapping
│   ├── tag.csv         # User tags
│   └── genome_tags.csv # Genome tag labels
├── frontend/
│   └── index.html      # Browser UI
├── models/
│   ├── knn_model.pkl   # Trained KNN model
│   ├── matrix.pkl      # Sparse movie-user ratings matrix
│   └── movie_index.pkl # Movie ID to matrix index mapping
├── notebook/
│   └── movie_recommender.ipynb
├── src/
│   ├── content_model.py
│   ├── data_loader.py
│   ├── model.py
│   ├── recommender.py
│   └── train.py
├── requirements.txt
└── README.md
```

## Dataset

This project uses MovieLens data.

Expected runtime/training files:

- `data/movie.csv`
- `data/rating.csv`
- `data/link.csv`

The backend uses `movie.csv` and `link.csv` at runtime. `rating.csv` is only needed for training.

## Environment Variables

Create a `.env` file in the project root:

```env
TMDB_API_KEY=your_tmdb_api_key_here
```

Without this key, recommendations still work, but poster and movie metadata requests will not return TMDB details.

## Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Training

Train or rebuild the model files:

```bash
python src/train.py
```

This writes model artifacts into `models/`.

Required output files for the Flask backend:

- `models/knn_model.pkl`
- `models/matrix.pkl`
- `models/movie_index.pkl`

`models/genre_sim.pkl` is not required by the current Flask backend. Genre similarity is computed from compact genre vectors at runtime to avoid loading a very large similarity matrix.

## Running the App

Start the backend:

```bash
python -m flask --app app.server run --host 127.0.0.1 --port 5000
```

Start the frontend in another terminal:

```bash
python -m http.server 8000 --directory frontend
```

Open:

```text
http://127.0.0.1:8000
```

## API

### Search Movies

```http
GET /search?q=toy
```

Example response:

```json
[
  "Toy Story (1995)",
  "Toy Story 2 (1999)"
]
```

### Get Recommendations

```http
POST /recommend
Content-Type: application/json
```

Request:

```json
{
  "movie": "Toy Story"
}
```

Example response:

```json
[
  {
    "title": "Star Wars: Episode IV - A New Hope (1977)",
    "tmdbId": 11
  }
]
```

### Get TMDB Details

```http
GET /movie/<tmdb_id>
```

Example:

```http
GET /movie/862
```

## Performance Notes

- Backend paths are resolved from the project root, so running from different directories is safer.
- The Flask API does not load `rating.csv` at runtime.
- The Flask API does not load the large `genre_sim.pkl` file at runtime.
- Search requests in the frontend are debounced.
- Old frontend search/recommend requests are aborted when newer input arrives.
- TMDB detail requests are cached in memory during the Flask process.

## Common Issues

### Backend Cannot Find Data Or Models

Check that these files exist:

```text
data/movie.csv
data/link.csv
models/knn_model.pkl
models/matrix.pkl
models/movie_index.pkl
```

If model files are missing, run:

```bash
python src/train.py
```

### Posters Or Metadata Do Not Load

Check `.env`:

```env
TMDB_API_KEY=your_tmdb_api_key_here
```

Restart the Flask backend after changing `.env`.

### Frontend Shows Network Error

Make sure both servers are running:

```text
Backend:  http://127.0.0.1:5000
Frontend: http://127.0.0.1:8000
```

## License

This project is for educational and research use. MovieLens data belongs to GroupLens. TMDB metadata belongs to TMDB and follows TMDB API terms.
