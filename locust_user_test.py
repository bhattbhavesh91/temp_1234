from locust import HttpUser, task, between, events
import time

class APITestUser(HttpUser):
    wait_time = between(0, 1)
    num_requests = 5  # Number of API calls per "priority" user

    def on_start(self):
        # Check if this user is the "priority" user
        if len(self.environment.runner.user_count) == 0:
            self.is_priority_user = True
            self.request_count = 0
        else:
            self.is_priority_user = False

    @task
    def call_api(self):
        if self.is_priority_user and self.request_count < self.num_requests:
            headers = {
                "Content-Type": "application/json"
            }
            payload = {
                # Add your API request payload here
            }
            with self.client.post("/your-api-endpoint", json=payload, headers=headers, catch_response=True) as response:
                if response.status_code == 200:
                    response.success()
                else:
                    response.failure(f"API call failed with status code: {response.status_code}")

            self.request_count += 1

            # Check if the priority user has made the required number of requests
            if self.request_count >= self.num_requests:
                self.environment.runner.quit()
        else:
            # Regular user request
            headers = {
                "Content-Type": "application/json"
            }
            payload = {
                # Add your API request payload here
            }
            with self.client.post("/your-api-endpoint", json=payload, headers=headers, catch_response=True) as response:
                if response.status_code == 200:
                    response.success()
                else:
                    response.failure(f"API call failed with status code: {response.status_code}")