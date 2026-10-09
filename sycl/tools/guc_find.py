import base64
import re
import sys

raw = open(sys.argv[1], 'rb').read().decode('latin1')
i = raw.index('[LOG].data: ')
j = raw.index('\n****', i)
enc = raw[i + len('[LOG].data: '):j].strip()
print('encoded chars:', len(enc))
try:
    blob = base64.a85decode(enc)
except Exception as e:
    print('a85decode failed:', e)
    blob = base64.a85decode(enc, adobe=False)
print('decoded bytes:', len(blob))

pats = {
    'LE32_hi': bytes.fromhex('924babea'),
    'LE48': bytes.fromhex('00a0924babea'),
    'LE64': bytes.fromhex('00a0924babeaffff'),
    'page_le': bytes.fromhex('2ab9b4aa'),
}
for name, p in pats.items():
    hits = [m.start() for m in re.finditer(re.escape(p), blob)]
    print('%s: %d hits' % (name, len(hits)), [hex(h) for h in hits[:6]])

h = blob.find(bytes.fromhex('924babea'))
if h >= 0:
    print('context around the first hit:')
    for off in range(h - 64, h + 96, 16):
        chunk = blob[off:off + 16]
        print('%08x  %-47s %s' % (off, ' '.join('%02x' % c for c in chunk),
                                  ''.join(chr(c) if 32 <= c < 127 else '.' for c in chunk)))
