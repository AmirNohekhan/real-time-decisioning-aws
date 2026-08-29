"""Run after starting the API; exits nonzero when the contract is unhealthy."""
import json
import urllib.request

payload = json.dumps({"user_id": "u00001", "k": 3, "context": {"device": "mobile", "hour": 20, "location_category": "urban", "session_depth": 2}}).encode()
request = urllib.request.Request("http://127.0.0.1:8000/v1/recommendations", data=payload, headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(request, timeout=5) as response:  # nosec B310: fixed localhost URL
    assert response.status == 200
    assert len(json.load(response)["recommendations"]) == 3

