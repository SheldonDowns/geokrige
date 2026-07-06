# geokrige

Fetch, test, and krige spatial point data — a small pipeline for turning
scattered sensor/station/listing observations into a smooth interpolated
surface, with the diagnostics to justify it along the way:

`fetch → aggregate → Moran's I → variogram fit → Kriging (auto OK/UK) → cross-validate → export`

Works with any point dataset that has a longitude, latitude, and a numeric
value — traffic counters, air quality monitors, real-estate prices, crime
incidents, soil samples, etc.

See [Quickstart](quickstart.md) to get started, or the
[API Reference](reference.md) for full module documentation.
