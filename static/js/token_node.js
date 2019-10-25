var TokenNode = function(parseTree, token) {
    var self = this;
    self.parseTree = parseTree;

    // import token data
    self.id = token.id;
    self.text = token.text;
    self.tag = token.tag;
    self.pos = token.pos;
    self.headId = token.head_id;
    self.dep = token.dep;
    self.nounChunkHead = token.noun_chunk_head;
    self.collapsedText = token.collapsed_text;
    self.childIds = token.child_ids;

    // collapse noun chunks, hide subtree if collapsed
    self.collapsed = token.noun_chunk_head;

    //   // collapse noun chunks, hide subtree if collapsed
    // self.collapsed = (
    //   token.noun_chunk_head ||
    //   token.dep == 'xcomp' ||
    //   token.dep == 'acomp' ||
    //   token.dep == 'ccomp' ||
    //   token.dep == 'advcl' ||
    //   token.dep == 'prep'
    // );

    // set head, children
    self.head = parseTree.getTokenNode(self.headId);
    self.children = [];

    if (self.dep == 'ROOT') {
        self.hidden = false;
    } else {
        self.head.children.push(self)
        self.hidden = self.head.collapsed || self.head.hidden
    }

    // color nouns pink and verbs blue
    if (self.pos == 'NOUN' || self.pos == 'PRON') {
        self.color = 'pink';
    } else if (self.pos == 'VERB') {
        self.color = 'lightblue';
    } else {
        self.color = 'lightgrey'
    }

    // UPDATE PARSETREE'S VISJS NETWORK
    self.render = function() {
        if (self.dep != 'ROOT') {
            self.hidden = self.head.collapsed || self.head.hidden;
        }

        self.parseTree.nodes.update({
            id: self.id,
            label: self.collapsed ? self.collapsedText : self.text,
            title: tagDescriptions[self.tag],
            color: self.color,
            hidden: self.hidden
        });
        self.parseTree.edges.update({
            id: self.id,
            from: self.headId,
            to: self.id,
            label: self.dep,
            title: depDescriptions[self.dep],
            arrows: 'to'
        });
    }
}
