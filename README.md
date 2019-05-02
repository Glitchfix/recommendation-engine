# recommendation-engine
A recommendation engine example using LightFM


To run:
`
python3 server.py
`

This will create a [Flask](https://github.com/pallets/flask) server running at `localhost:5000`
You can pick a user ID between the given range and then get its known positives and the recommended movies that you must watch.
This example is inspired from [LightFM](https://github.com/lyst/lightfm) library for the factorization of the coordinate list sparse matrix using WARP loss function.
