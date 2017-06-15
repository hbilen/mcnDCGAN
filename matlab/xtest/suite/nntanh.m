classdef nntanh < nntest
  methods (Test)

    function basic(test)
      batchSize = 10 ;
      x = test.randn([5 5 3 batchSize]) ;
      y = vl_nntanh(x) ;

      % check derivatives with numerical approximation
      dzdy = test.randn(size(y)) ;
      derInputs = vl_nntanh(x, dzdy) ;
      dzdx = derInputs{1} ;
      test.der(@(x) vl_nntanh(x), x, dzdy, dzdx, 1e-5*test.range) ;
    end

  end
end
