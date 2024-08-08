from locust import HttpUser, task, events
from collections import defaultdict

class MyUser(HttpUser):
    wait_time = between(1, 3)
    user_request_counts = defaultdict(int)

    @task
    def hit_api(self):
        user_id = self.user_id
        if self.user_request_counts[user_id] < 5:
            with self.client.post("/your-api-endpoint", catch_response=True) as response:
                self.user_request_counts[user_id] += 1
                if response.status_code != 200:
                    response.failure("API request failed")

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    environment.user_request_counts = defaultdict(int)

@events.request.add_listener
def on_request(request_type, name, response_time, response_length, response, context, **kwargs):
    user_id = context.user.user_id
    environment.user_request_counts[user_id] += 1

# Run the test:
# locust -f locust_script.py --host=http://your-api-url.com