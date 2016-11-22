var Widget = function(){
    var self = Core({
            data: {
                message: 'This is the detail',
                seen: true
            },
            methods: {
               toggling: function(){
                  this.seen = false;
                  this.message = ''
               },
               re_toggling: function(){
                  this.seen = true;
                  this.message = 'This is the detail'
               }
            }
        });
   return self;
};
var WIDGET = null;
jQuery(function(){WIDGET = Widget();});
