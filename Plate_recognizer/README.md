### Load Weights File
- load the [weight.zip](https://drive.google.com/drive/u/0/folders/1PpLbmgrrQS7ssrItZXHsyO-1eWHeYC2H) and extract to `/models`

### Start Local
1. `docker network create alpr-network`
2. `docker compose up --build`

### Run on CPU
- `docker compose -f compose.cpu.yml up --build`

### Return Format
| **Field**          | **Type**                                    | **Default** |
|---------------------|--------------------------------------------|-------------|
| `car_bbox`         | `list` of `(xmin, ymin, xmax, ymax)`        | `None`      |
| `plate_bbox`       | `list` of `(xmin, ymin, xmax, ymax)`        | `None`      |
| `text_bbox_list`   | `list` of `list` `(xmin, ymin, xmax, ymax)` | `None`      |
| `plate_id`         | `string`                                    | `None`      |
| `province`         | `string`                                    | `None`      |
| `full_plate`       | `string`                                    | `None`      |
| `format_flag`      | `string` (`complete` \| `warning`)          | `warning`   |
| `message`          | `string`                                    | `''`        |
