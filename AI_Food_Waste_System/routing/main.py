"""
AI-Powered Smart Food Waste Reduction Ecosystem
Routing & Logistics Microservice — Sandbox Mode

This FastAPI service implements a "publish-and-claim" marketplace model:
  1. A food processing unit publishes surplus → we broadcast to nearby NGOs.
  2. An NGO accepts → they book a third-party delivery through our platform.
  3. The delivery partner posts webhook status updates → we relay them.

SANDBOX BEHAVIOUR:
  • The /broadcast endpoint keeps the original sandbox behaviour and
    simulates driving times with random values (15–90 min).
  • The /distances endpoint uses the public OSRM routing service to
    calculate road distance and estimated driving time between coordinates.
  • No real delivery API is called.  Quotes, delivery IDs, costs, and
    ETAs are still generated with uuid / random for sandbox development.
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timezone
from typing import Optional
import httpx

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Food Waste Routing & Logistics — Sandbox",
    version="1.0.0",
    description=(
        "Sandbox microservice that mocks mapping & delivery APIs "
        "for the AI-Powered Smart Food Waste Reduction Ecosystem."
    ),
)

# Allow all origins during development; tighten in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic data models
# ---------------------------------------------------------------------------


class SurplusListing(BaseModel):
    """Payload sent by a food processing unit when publishing surplus."""

    pickup_lat: float = Field(..., ge=-90, le=90, description="Pickup latitude")
    pickup_lng: float = Field(..., ge=-180, le=180, description="Pickup longitude")
    expiry_time_mins: int = Field(
        ..., gt=0, description="Minutes until the food expires"
    )


class NGOProfile(BaseModel):
    """Represents a registered NGO that can claim surplus food."""

    id: str
    name: str
    lat: float
    lng: float


class EligibleNGO(BaseModel):
    """An NGO that passed the time-feasibility filter, returned to the caller."""

    id: str
    name: str
    lat: float
    lng: float
    simulated_drive_time_mins: int
    time_remaining_after_buffer_mins: int


class BroadcastResponse(BaseModel):
    """Response from the /broadcast endpoint."""

    pickup_lat: float
    pickup_lng: float
    expiry_time_mins: int
    eligible_ngos: list[EligibleNGO]
    total_ngos_checked: int
    message: str

class LocationPoint(BaseModel):
    """A receiver destination used for routing."""

    id: str
    name: str
    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)


class DistanceRequest(BaseModel):
    """Provider location and receiver destinations."""

    origin_lat: float = Field(..., ge=-90, le=90)
    origin_lng: float = Field(..., ge=-180, le=180)
    destinations: list[LocationPoint]


class DistanceResult(BaseModel):
    """Road distance and travel time for one destination."""

    id: str
    name: str
    lat: float
    lng: float
    distance_km: Optional[float] = None
    duration_mins: Optional[int] = None


class DistanceResponse(BaseModel):
    """Routing results returned to FoodLoop."""

    origin_lat: float
    origin_lng: float
    results: list[DistanceResult]

class DeliveryRequest(BaseModel):
    """Payload sent by an NGO (via the frontend) to book a delivery."""

    ngo_id: str
    pickup_lat: float = Field(..., ge=-90, le=90)
    pickup_lng: float = Field(..., ge=-180, le=180)
    dropoff_lat: float = Field(..., ge=-90, le=90)
    dropoff_lng: float = Field(..., ge=-180, le=180)


class DeliveryBookingResponse(BaseModel):
    """Simulated response from a third-party delivery partner (e.g. Porter)."""

    delivery_id: str
    quote_id: str
    ngo_id: str
    estimated_cost_inr: float
    estimated_eta_mins: int
    tracking_url: str
    provider: str
    status: str
    booked_at: str


class WebhookPayload(BaseModel):
    """Payload posted by the delivery partner's webhook."""

    delivery_id: str
    status: str = Field(
        ...,
        description="e.g. driver_assigned, pickup_reached, in_transit, delivered",
    )
    timestamp: Optional[str] = None
    driver_name: Optional[str] = None
    driver_phone: Optional[str] = None


class WebhookAck(BaseModel):
    """Acknowledgement returned to the delivery partner."""

    received: bool
    delivery_id: str
    status: str
    message: str


# ---------------------------------------------------------------------------
# Mock NGO database
# ---------------------------------------------------------------------------
# In production this would come from a real database.  For the sandbox we
# hard-code five NGOs spread across different parts of a city so the random
# driving-time simulation produces a realistic mix of eligible / ineligible
# results.

MOCK_NGOS: list[NGOProfile] = [
    NGOProfile(id="ngo-001", name="Feeding India Foundation", lat=28.6139, lng=77.2090),
    NGOProfile(id="ngo-002", name="Robin Hood Army Delhi", lat=28.5355, lng=77.3910),
    NGOProfile(id="ngo-003", name="Goonj Warehouse", lat=28.4595, lng=77.0266),
    NGOProfile(id="ngo-004", name="Akshaya Patra NCR", lat=28.7041, lng=77.1025),
    NGOProfile(id="ngo-005", name="No Food Waste Gurugram", lat=28.4089, lng=77.3178),
]

# In-memory store for booked deliveries (sandbox only).
DELIVERY_STORE: dict[str, dict] = {}

# Buffer time (in minutes) added to driving time as a safety margin.
BUFFER_MINS = 30

# ---------------------------------------------------------------------------
# Helper — simulate driving time
# ---------------------------------------------------------------------------


def _simulate_drive_time() -> int:
    """Return a random driving time between 15 and 90 minutes.

    SANDBOX LOGIC:
    In production, this would call Google Maps Distance Matrix API (or a
    self-hosted OSRM / Valhalla instance) with the pickup and NGO coordinates.
    Here we simply draw a uniform random integer to keep the service free of
    external dependencies.
    """
    return random.randint(15, 90)


# ---------------------------------------------------------------------------
# Endpoint 1 — Smart Broadcasting
# ---------------------------------------------------------------------------


@app.post(
    "/api/logistics/broadcast",
    response_model=BroadcastResponse,
    summary="Smart broadcast surplus to nearby, time-feasible NGOs",
    tags=["Logistics"],
)
async def broadcast_surplus(listing: SurplusListing):
    """Simulate the 'Smart Broadcasting' step.

    For each NGO in the mock database:
      1. Generate a random driving time (mocking a real mapping API).
      2. Check feasibility: drive_time + 30-min buffer ≤ expiry_time_mins.
      3. Return only the NGOs that pass the filter.

    The frontend should display these eligible NGOs so one of them can
    accept and proceed to book a delivery.
    """
    eligible: list[EligibleNGO] = []

    for ngo in MOCK_NGOS:
        drive_time = _simulate_drive_time()
        total_required = drive_time + BUFFER_MINS

        if total_required <= listing.expiry_time_mins:
            eligible.append(
                EligibleNGO(
                    id=ngo.id,
                    name=ngo.name,
                    lat=ngo.lat,
                    lng=ngo.lng,
                    simulated_drive_time_mins=drive_time,
                    time_remaining_after_buffer_mins=(
                        listing.expiry_time_mins - total_required
                    ),
                )
            )

    # Sort by shortest drive time so the best candidates appear first.
    eligible.sort(key=lambda n: n.simulated_drive_time_mins)

    if eligible:
        message = f"{len(eligible)} NGO(s) can reach pickup before expiry."
    else:
        message = (
            "No NGOs can reach the pickup in time. "
            "Consider increasing the expiry window or broadening the search."
        )

    return BroadcastResponse(
        pickup_lat=listing.pickup_lat,
        pickup_lng=listing.pickup_lng,
        expiry_time_mins=listing.expiry_time_mins,
        eligible_ngos=eligible,
        total_ngos_checked=len(MOCK_NGOS),
        message=message,
    )


# ---------------------------------------------------------------------------
# Endpoint 2 — Road distance and travel-time calculation
# ---------------------------------------------------------------------------


@app.post(
    "/api/logistics/distances",
    response_model=DistanceResponse,
    summary="Calculate road distance and travel time to receiver locations",
    tags=["Logistics"],
)
async def calculate_distances(req: DistanceRequest):
    """Calculate road distance and estimated driving time from one origin.

    The origin is normally the food provider's current location. Each
    destination represents a receiver/NGO location.

    OSRM expects coordinates in longitude,latitude order. Results are sorted
    from nearest to farthest by road distance.
    """

    if not req.destinations:
        return DistanceResponse(
            origin_lat=req.origin_lat,
            origin_lng=req.origin_lng,
            results=[],
        )

    # OSRM coordinate order is longitude,latitude.
    coordinates = [
        f"{req.origin_lng},{req.origin_lat}"
    ]

    for destination in req.destinations:
        coordinates.append(
            f"{destination.lng},{destination.lat}"
        )

    coordinate_string = ";".join(coordinates)

    # First coordinate is the source; all remaining coordinates are destinations.
    destination_indexes = ";".join(
        str(index)
        for index in range(1, len(coordinates))
    )

    url = (
        "https://router.project-osrm.org/"
        f"table/v1/driving/{coordinate_string}"
    )

    params = {
        "sources": "0",
        "destinations": destination_indexes,
        "annotations": "distance,duration",
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                url,
                params=params,
                headers={
                    "User-Agent": "FoodLoopAI/1.0"
                },
            )

        response.raise_for_status()

        data = response.json()

        if data.get("code") != "Ok":
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=(
                    "Routing service could not calculate "
                    "the requested routes."
                ),
            )

        distances = data.get("distances", [[]])[0]
        durations = data.get("durations", [[]])[0]

        if len(distances) != len(req.destinations) or len(durations) != len(req.destinations):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Routing service returned an incomplete route matrix.",
            )

        results: list[DistanceResult] = []

        for index, destination in enumerate(req.destinations):
            distance_meters = (
                distances[index]
                if index < len(distances)
                else None
            )

            duration_seconds = (
                durations[index]
                if index < len(durations)
                else None
            )

            results.append(
                DistanceResult(
                    id=destination.id,
                    name=destination.name,
                    lat=destination.lat,
                    lng=destination.lng,
                    distance_km=(
                        round(distance_meters / 1000, 2)
                        if distance_meters is not None
                        else None
                    ),
                    duration_mins=(
                        round(duration_seconds / 60)
                        if duration_seconds is not None
                        else None
                    ),
                )
            )

        # Nearest receiver first.
        results.sort(
            key=lambda item: (
                item.distance_km
                if item.distance_km is not None
                else float("inf")
            )
        )

        return DistanceResponse(
            origin_lat=req.origin_lat,
            origin_lng=req.origin_lng,
            results=results,
        )

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "Routing service returned an HTTP error: "
                f"{exc.response.status_code}"
            ),
        ) from exc

    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Routing service unavailable: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# Endpoint 3 — Book a mock third-party delivery
# ---------------------------------------------------------------------------


@app.post(
    "/api/logistics/book-delivery",
    response_model=DeliveryBookingResponse,
    summary="Book a simulated third-party delivery (mock Porter / Uber Direct)",
    tags=["Logistics"],
)
async def book_delivery(req: DeliveryRequest):
    """Simulate calling a third-party delivery API.

    SANDBOX LOGIC:
    In production, this would POST to the Porter or Uber Direct REST API
    with the pickup / dropoff coordinates and return a real quote.  Here
    we generate fake IDs, a random cost (₹80–₹350) and a random ETA
    (20–60 min).

    The delivery record is stored in an in-memory dict so the webhook
    endpoint can reference it later.
    """
    # Validate that the NGO exists in our mock database.
    ngo_match = next((n for n in MOCK_NGOS if n.id == req.ngo_id), None)
    if ngo_match is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"NGO with id '{req.ngo_id}' not found in the database.",
        )

    delivery_id = f"DEL-{uuid.uuid4().hex[:12].upper()}"
    quote_id = f"QUO-{uuid.uuid4().hex[:10].upper()}"
    cost = round(random.uniform(80, 350), 2)
    eta = random.randint(20, 60)
    tracking_url = f"https://sandbox.porter.in/track/{delivery_id}"
    booked_at = datetime.now(timezone.utc).isoformat()

    booking = DeliveryBookingResponse(
        delivery_id=delivery_id,
        quote_id=quote_id,
        ngo_id=req.ngo_id,
        estimated_cost_inr=cost,
        estimated_eta_mins=eta,
        tracking_url=tracking_url,
        provider="Porter (Sandbox)",
        status="confirmed",
        booked_at=booked_at,
    )

    # Persist in sandbox store so webhooks can reference this delivery.
    DELIVERY_STORE[delivery_id] = booking.model_dump()

    return booking


# ---------------------------------------------------------------------------
# Endpoint 4 — Delivery webhook listener (live tracking)
# ---------------------------------------------------------------------------


@app.post(
    "/api/webhooks/delivery",
    response_model=WebhookAck,
    summary="Receive delivery status updates from the logistics partner",
    tags=["Webhooks"],
)
async def delivery_webhook(payload: WebhookPayload):
    """Listen for status webhooks from the delivery partner.

    SANDBOX LOGIC:
    In production this would:
      • Validate an HMAC signature from the delivery partner.
      • Update the delivery status in a persistent database.
      • Push the update to connected frontends via WebSocket / SSE.
    Here we simply log to stdout and return 200 OK.
    """
    timestamp = payload.timestamp or datetime.now(timezone.utc).isoformat()

    # ── Console log (simulates pushing to a frontend WebSocket) ──
    print("=" * 60)
    print("📦  DELIVERY WEBHOOK RECEIVED")
    print(f"    Delivery ID : {payload.delivery_id}")
    print(f"    Status      : {payload.status}")
    print(f"    Timestamp   : {timestamp}")
    if payload.driver_name:
        print(f"    Driver      : {payload.driver_name}")
    if payload.driver_phone:
        print(f"    Phone       : {payload.driver_phone}")
    print("=" * 60)

    # Update in-memory store if the delivery exists.
    if payload.delivery_id in DELIVERY_STORE:
        DELIVERY_STORE[payload.delivery_id]["status"] = payload.status

    return WebhookAck(
        received=True,
        delivery_id=payload.delivery_id,
        status=payload.status,
        message=f"Status '{payload.status}' acknowledged.",
    )


# ---------------------------------------------------------------------------
# Health-check
# ---------------------------------------------------------------------------


@app.get("/health", tags=["Health"])
async def health_check():
    """Simple liveness probe."""
    return {
        "service": "routing-logistics-sandbox",
        "status": "healthy",
        "mock_ngos_loaded": len(MOCK_NGOS),
        "active_deliveries": len(DELIVERY_STORE),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
