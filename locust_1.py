from locust import HttpUser, task, between, events

class APITestUser(HttpUser):
    wait_time = between(0, 1)
    num_requests = 5  # Number of API calls per user

    def on_start(self):
        self.request_count = 0
        self.user_id = id(self)

    @task
    def call_api(self):
        if self.request_count < self.num_requests:
            headers = {
                "Content-Type": "application/json"
            }
            payload = {
                # Add your API request payload here
            }
            with self.client.post("/your-api-endpoint", json=payload, headers=headers, catch_response=True) as response:
                if response.status_code == 200:
                    events.request_success.fire(
                        request_type="POST",
                        name="/your-api-endpoint",
                        response_time=response.elapsed.total_seconds() * 1000,
                        response_length=len(response.content),
                        user_id=self.user_id
                    )
                else:
                    events.request_failure.fire(
                        request_type="POST",
                        name="/your-api-endpoint",
                        response_time=response.elapsed.total_seconds() * 1000,
                        exception=f"API call failed with status code: {response.status_code}",
                        user_id=self.user_id
                    )

            self.request_count += 1

            # Check if the user has made the required number of requests
            if self.request_count >= self.num_requests:
                self.environment.runner.quit()

user_request_counts = {}

@events.request_success.add_listener
def on_request_success(request_type, name, response_time, response_length, **kwargs):
    user_id = kwargs["user_id"]
    if user_id not in user_request_counts:
        user_request_counts[user_id] = 0
    user_request_counts[user_id] += 1

@events.request_failure.add_listener
def on_request_failure(request_type, name, response_time, exception, **kwargs):
    user_id = kwargs["user_id"]
    if user_id not in user_request_counts:
        user_request_counts[user_id] = 0
    user_request_counts[user_id] += 1

@events.test_stop.add_listener
def test_stop(environment, **kwargs):
    total_users = len(user_request_counts)
    total_requests = sum(user_request_counts.values())

    for user_id, requests_made in user_request_counts.items():
        print(f"User ID: {user_id}, Requests made: {requests_made}")

    print(f"Total number of users: {total_users}")
    print(f"Total number of requests: {total_requests}")