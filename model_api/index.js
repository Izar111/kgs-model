const { ApolloServer, gql } = require('apollo-server');
const child_process = require('child_process');

const call_process = child_process.spawn
let result_str = ''
let result_json = {}

// A schema is a collection of type definitions (hence "typeDefs")
// that together define the "shape" of queries that are executed against
// your data.
const typeDefs = gql`
  # Comments in GraphQL strings (such as this one) start with the hash (#) symbol.

  # This "Book" type defines the queryable fields for every book in our data source.
  type Book {
    title: String
    author: String
  }

  # The "Query" type is special: it lists all of the available queries that
  # clients can execute, along with the return type for each. In this
  # case, the "books" query returns an array of zero or more Books (defined above).
  type Query {
    books: [Book]
  }
`;

// Resolvers define the technique for fetching the types defined in the
// schema. This resolver retrieves books from the "books" array above.
const resolvers = {
  Query: {
  },
};

// The ApolloServer constructor requires two parameters: your schema
// definition and your set of resolvers.
const server = new ApolloServer({
  typeDefs,
  resolvers,
  csrfPrevention: true,
});

//model call function
function call_model(args){

  const test = call_process('python3',['model_run.py',args]);
  
      
  //model result get
  test.stdout.on('data', function(data) { 
    result_str += data.toString(); 
  } ); 
  //check error
  test.stderr.on('data', function(data) { console.log(data.toString()); });
  //model is end
  test.on('exit',function() { console.log(result_str); result_json = JSON.parse(result_str); } )
}

// The `listen` method launches a web server.
server.listen({port: 8080,}).then(({ url }) => {
  
  console.log(`๐  Server ready at ${url}`);
  
  call_model("{'sentence':['์ํฅ๋ฏผ์ ๋ํ๋ฏผ๊ตญ์ ์ถ๊ตฌ์ ์์ด๋ค .','์ ๋กฌ ํ์ ๋ฏธ๊ตญ ์ฐ๋ฐฉ์ค๋น์ ๋(Fedยท์ฐ์ค) ์์ฅ์ 26์ผ(ํ์ง์๊ฐ) ๊ธฐ์ค๊ธ๋ฆฌ๋ฅผ ์ฌ๋ฆด ์ฌ๋ ฅ์ด ์ถฉ๋ถํ๋ค๋ ์์ฅ์ ๋ฐํ๋ค.'],'link':['aaa','bbb']}")
  //call_model();

});
