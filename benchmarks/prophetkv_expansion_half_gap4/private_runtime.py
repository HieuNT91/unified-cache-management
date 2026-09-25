"""Local-only variable fresh suffix adaptation, preserving 256-token arithmetic."""
from pathlib import Path


def adapt(private):
    directory=Path(private)/'sparse/prophetkv'
    p=directory/'selection.py';text=p.read_text()
    assert 'b[-1]-b[-2]!=256' in text
    p.write_text(text.replace('b[-1]-b[-2]!=256','b[-1]-b[-2]<256').replace(
        'context and 256 fresh tokens','context and at least 256 fresh tokens'))
    p=directory/'runtime.py';text=p.read_text()
    text=text.replace('    b=meta.boundaries\n','    b=meta.boundaries\n    suffix_tokens=b[-1]-b[-2]\n')
    for a,b in [('-256:','-suffix_tokens:'),('causal_lower_right(256,','causal_lower_right(suffix_tokens,'),
                ('reshape(256,-1)','reshape(suffix_tokens,-1)'),('suffix_tokens=256','suffix_tokens=suffix_tokens'),
                ('probe_tokens_per_layer=256','probe_tokens_per_layer=suffix_tokens')]:
        assert a in text,a
        text=text.replace(a,b)
    p.write_text(text)
    p=directory/'prophetkv.py';text=p.read_text()
    for a,b in [('fresh_suffix_tokens=256','fresh_suffix_tokens=self.request.boundaries[-1]-self.request.boundaries[-2]'),
                ('[:-256]','[:-(self.request.boundaries[-1]-self.request.boundaries[-2])]'),
                ('[-256:]','[-(self.request.boundaries[-1]-self.request.boundaries[-2]):]')]:
        assert a in text,a
        text=text.replace(a,b)
    p.write_text(text)
