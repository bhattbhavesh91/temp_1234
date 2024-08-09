import time
from locust import HttpUser, task, between
from concurrent.futures import ThreadPoolExecutor, as_completed

class ApiUser(HttpUser):
    wait_time = between(1, 5)  # Wait 1-5 seconds between tasks

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executor = ThreadPoolExecutor(max_workers=5)

    @task
    def test_api(self):
        futures = []
        for i in range(1, 6):
            futures.append(self.executor.submit(self.make_request, f"/route{i}"))

        results = []
        for future in as_completed(futures):
            results.append(future.result())

        # Process the final results here
        self.process_results(results)

    def make_request(self, route):
        payload = {"data": f"Data for {route}"}
        response = self.client.post(route, json=payload)
        return response.json()

    def process_results(self, results):
        # Process the results from all 5 routes
        # This is where you'd implement your logic to handle the final output
        combined_result = " ".join([result.get("message", "") for result in results])
        print(f"Final result: {combined_result}")