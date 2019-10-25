var ParseTree = function() {
    var self = this;

    self.nodes = new vis.DataSet();
    self.edges = new vis.DataSet();

    var container = document.getElementById('parse-tree');
    var data = {nodes: self.nodes, edges: self.edges};
    var options = {};
    self.network = new vis.Network(container, data, options);

    self.tokenNodes = []

    self.fit = function() {
        var fitOptions = {
            nodes: self.nodes.getIds()
        };
        self.network.fit(fitOptions);
    };

    //// DOESN'T BELONG
    // RETURN TOKENNODE BY ID
    self.getTokenNode = function(id) {
        var match = null;
        self.tokenNodes.forEach(function(tokenNode) {
            if (tokenNode.id == id) match = tokenNode;
        });
        return match;
    }

    // GENERATE TOKENNODES FROM TOKENS, render
    self.importSubtree = function(tokens, token) {
        tokenNode = new TokenNode(self, token);
        self.tokenNodes.push(tokenNode);
        token.child_ids.forEach(function(childId) {
            self.importSubtree(tokens, tokens[childId])
        });
    }

    self.renderSubtree = function(tokenNode) {
        tokenNode.render();
        tokenNode.children.forEach(function(child) {
            self.renderSubtree(child);
        });
    }

    //// BELONS HERE
    // PROCESS QUERY, DISPLAY RESULTS
    self.render = function(tokens) {
        // console.log('ParseTree: rendering...')
        self.tokenNodes = []
        tokens.forEach(function(token) {
            if (token.dep == 'ROOT') {
                self.importSubtree(tokens, token);
                self.renderSubtree(self.getTokenNode(token.id));
            }
        });

        // remove additional nodes
        self.nodes.getIds().forEach(function(id) {
            if (!self.getTokenNode(id)) {
                self.nodes.remove(id);
            }
        });
        // console.log('ParseTree: render complete.')
    };

    //// BELONGS HERE
    // COLLAPSE SUBTREE WHEN CLICKED
    self.network.on('click', function(properties) {
        if (properties.nodes.length > 0) {
            tokenNode = self.getTokenNode(properties.nodes[0]);
            tokenNode.collapsed = !tokenNode.collapsed;
            self.renderSubtree(tokenNode);
        }
    });
}
