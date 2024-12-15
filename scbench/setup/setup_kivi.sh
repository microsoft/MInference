# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

git clone https://github.com/jy-yuan/KIVI.git /tmp/KIVI
cd /tmp/KIVI
pip install -e . --no-build-isolation --no-deps
cd quant && pip install -e .
