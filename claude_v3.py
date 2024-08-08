from locust import HttpUser, task, between, events
from collections import defaultdict

total_requests = 0
total_users = 0

class MyUser(HttpUser):
    wait_time = between(1, 5)
    
    @task
    def post_request(self):
        global total_requests, total_users
        
        # Check if the user has reached the request limit
        if len(self.client.request_history) >= 5:
            print(f"User {self.user_id} has reached the request limit of 5.")
            return
        
        # Make the POST request
        self.client.post("/your-api-endpoint", json={"data": "some_data"})
        
        total_requests += 1
        total_users += 1
        
    def on_stop(self):
        global total_requests, total_users
        print(f"Total requests: {total_requests}")
        print(f"Total users: {total_users}")

if __name__ == "__main__":
    events.quitting += lambda stats: print(f"Total requests: {total_requests}")
    events.quitting += lambda stats: print(f"Total users: {total_users}")
    
    # Start the load testing
    locust --host=http://your-api-host.com -f locust_script.py