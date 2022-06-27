# Get  NHL Twitter Handles

using Lazy

things_url = "https://gist.githubusercontent.com/torvarun/a99c7056b0d5490dbeef0b8d70199a6c/raw/7d4961f6e8ada6364f5ad0e29bc6f0be9d86f44e/nhltwitter"
raw = read(download(things_url), String)
# the @>> macro treats the value of the prior function as the *last* argument of the next function
things = @>> begin
    findall(r"@\w+,/?", raw)
    map(idx-> raw[idx])
    map(w->strip(w,['@',',']))
    #map(w->split(w,"@"))
    #map(last)
    filter(!isempty)
    unique
    sort
    ws-> join(ws,'\n')
end

fd = open("handles_nhl.csv","w")
print(fd,"handle\n")
print(fd,things)
close(fd)
