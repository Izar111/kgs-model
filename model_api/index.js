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

function call_model(args){

  const test = call_process('python3',['model_run.py',args]);
  const recode_start = ">>>End NER-RE=================";

  var rec = false;
  
  //test.stdout.on('data', function(data) { result_str = data.toString(); console.log("+++++++++++++++++++++++"); console.log(data.toString()); console.log("+++++++++++++++++++++++"); } ); 
  test.stdout.on('data', function(data) {     
    result_str += data.toString(); 
  } ); 
  test.stderr.on('data', function(data) { console.log(data.toString()); });
  //test.on('exit',function() { result_json = JSON.parse(result_str); console.log("result is \n",result_json); } )
  test.on('exit',function() { console.log(result_str) } )
}

// The `listen` method launches a web server.
server.listen({port: 8080,}).then(({ url }) => {
  
  console.log(`🚀  Server ready at ${url}`);
  
  call_model("{'sentence':['손흥민은 대한민국의 축구선수이다 .','제롬 파월 미국 연방준비제도(Fed·연준) 의장은 26일(현지시간) 기준금리를 올릴 여력이 충분하다는 입장을 밝혔다.'],'link':['aaa','bbb']}")

});
