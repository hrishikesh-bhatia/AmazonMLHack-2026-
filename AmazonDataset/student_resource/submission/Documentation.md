# Amazon ML Challenge 2026 - Business Entity Resolution

## Approach
Used Unicode normalization and token-based candidate generation with exact-name and
exact-address matching plus informative-token blocking.

## Matching
Candidates were selected using normalized business-name and address signals.

## Reproducibility
The solution code is included in code/business_entity_resolution/.
